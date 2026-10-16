"""Watches the container's memory while a variant runs, so a server that fills it is ended by the harness while the
container and its logs still exist. RunPod stops a container at about 98% of its memory limit and removes it, and
every file on it goes too; vllm#56771 attempt 6 lost its server.log, and any trace of what filled the memory, that way.

The first line of <run_dir>/memory.jsonl, controls, says whether the cgroup's memory.reclaim and memory.high exist and
are writable here. Then every MEM_GUARD_POLL_S seconds (default 5) one line: t, current, max (null for no limit), and
the memory.stat counters that tell anonymous memory, page cache, shmem, and locked pages apart. Each poll also follows
the server's process tree from <run_dir>/server.pid (serve_and_smoke.sh writes it) through /proc/*/stat.

At MEM_GUARD_TRIM_FRACTION of the limit (default 0.80), at most once a minute, the guard trims: it drops the cache
paths from the page cache, then asks memory.reclaim for the bytes over the trim line, at most 4 GiB, when it can and
the usage is under the kill line. The line carries trim (fadvise: whether that pass could run; reclaim: ok,
unavailable, skipped, or an errno name) and after_trim. Once a trim frees under 1 GiB, trims stop until the usage
reaches the kill line. memory.high and memory.max are never written.

When the usage after any trim of that poll is at MEM_GUARD_KILL_FRACTION (default 0.93) or over, and server.pid
exists, the guard fires: it
stops (SIGSTOP) every process of the server's tree, adds a mem_guard phase to <run_dir>/progress.jsonl, writes
<run_dir>/memory_guard.txt (its reading on the first line, then memory.stat, the largest processes, and each server
process's memory lines, threads, and kernel stacks, then py-spy stacks within 15 s), kills (SIGKILL) the tree, and
drops the cache paths again once the server's mappings are gone (an after_kill line). Only when server.pid holds no
pid does it fall back to `pkill -9 -f <server_pattern>`. It then keeps logging until stopped, and never fires twice.
Under no limit, or one it cannot read, it only logs.

Usage: mem_guard.py <run_dir> <parent_pid> <server_pattern> [<cache_path>...]   (exits once <parent_pid> is gone)
       mem_guard.py --drop-cache <path>...   (one summary line)
       mem_guard.py --limit                   (the limit in bytes, or none, then where it came from)
  MEM_GUARD_CGROUP (default /sys/fs/cgroup) is the cgroup v2 root, or the v1 root that holds memory/; under it the
  path /proc/self/cgroup names comes first. MEM_GUARD_PROC (default /proc) is where processes are read.
"""

import collections
import datetime
import errno
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

COUNTERS = ("anon", "file", "active_file", "inactive_file", "shmem", "file_mapped", "file_dirty", "unevictable")
# cgroup v1 names for the v2 counters that differ. Its total_ forms count child cgroups too, as v2's counters do.
V1_NAMES = {"anon": "rss", "file": "cache", "file_mapped": "mapped_file", "file_dirty": "dirty"}
# v2 writes no limit as "max", v1 as the largest page-aligned 64-bit value.
UNLIMITED = 2**62
GIB = 2**30
TRIM_EVERY_S = 60
# A trim that frees less than this found the cache mapped (vLLM maps the checkpoint while it copies the weights), and
# repeating a full pass every minute then only slows the server down.
USEFUL_TRIM = GIB
# The kernel reclaims in the writer's context, so a large request holds up the poll for as long as it takes.
RECLAIM_CAP = 4 * GIB
STACK_BUDGET_S = 15
SWEEP_TIMEOUT_S = 5
EXIT_WAIT_S = 10
TABLE_LINES = 40
STATUS_KEYS = ("State", "VmRSS", "VmLck", "VmPin", "RssAnon", "RssFile", "RssShmem")
CONTROLS = ("memory.reclaim", "memory.high")
PROC = os.environ.get("MEM_GUARD_PROC", "/proc")

Stat = collections.namedtuple("Stat", "ppid state start rss comm")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message):
    print(f"{utc_now()} {message}", file=sys.stderr, flush=True)


def append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def knob(name, default):
    value = os.environ.get(name, "")
    try:
        return float(value) if value else default
    except ValueError:
        log(f"{name}={value!r} is not a number; using {default}")
        return default


def cgroup_root():
    return os.environ.get("MEM_GUARD_CGROUP", "/sys/fs/cgroup")


def own_cgroups():
    """(v2 path, v1 memory path) of this process, from /proc/self/cgroup; either may be None."""
    v2 = v1 = None
    try:
        text = read_text(os.path.join(PROC, "self", "cgroup"))
    except OSError:
        return None, None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and not parts[1]:
            v2 = parts[2]
        elif "memory" in parts[1].split(","):
            v1 = parts[2]
    return v2, v1


def candidates(root, path):
    # A container in the host's cgroup namespace sees the host's root here and its own cgroup at the path
    # /proc/self/cgroup names; one in its own namespace sees its cgroup at the root and that path again.
    nested = [os.path.join(root, path.lstrip("/"))] if path and path != "/" else []
    return nested + [root]


def cgroup_files(root):
    """The cgroup directory, its usage, limit, and stat files, and the memory.stat names to try for each counter: v2
    when present, else v1."""
    v2, v1 = own_cgroups()
    for directory in candidates(root, v2):
        if os.path.exists(os.path.join(directory, "memory.current")):
            files = [os.path.join(directory, name) for name in ("memory.current", "memory.max", "memory.stat")]
            return directory, files, {key: [key] for key in COUNTERS}
    for directory in candidates(os.path.join(root, "memory"), v1):
        if os.path.exists(os.path.join(directory, "memory.usage_in_bytes")):
            files = [os.path.join(directory, name)
                     for name in ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.stat")]
            names = {key: ["total_" + V1_NAMES.get(key, key), V1_NAMES.get(key, key)] for key in COUNTERS}
            return directory, files, names
    files = [os.path.join(root, name) for name in ("memory.current", "memory.max", "memory.stat")]
    return root, files, {key: [key] for key in COUNTERS}


def current_usage(root):
    return int(read_text(cgroup_files(root)[1][0]))


def control(directory, name):
    """missing, read-only (container runtimes often mount the cgroup tree so), or writable."""
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return "missing"
    return "writable" if os.access(path, os.W_OK) else "read-only"


class Reading:
    """One look at the cgroup: usage, limit (None for none, with no_limit saying why), the logged counters, and
    memory.stat as the kernel wrote it."""

    def __init__(self, root):
        self.directory, (usage, self.limit_file, stat_file), names = cgroup_files(root)
        self.t = utc_now()
        self.current = int(read_text(usage))
        try:
            self.stat_text = read_text(stat_file)
        except OSError as e:
            self.stat_text = f"{stat_file}: {e}\n"
        values = {}
        for line in self.stat_text.splitlines():
            words = line.split()
            if len(words) == 2 and words[1].isdigit():
                values[words[0]] = int(words[1])
        self.counters = {key: next((values[n] for n in names[key] if n in values), None) for key in COUNTERS}
        self.limit, self.no_limit = None, None
        try:
            raw = read_text(self.limit_file).strip()
            if raw == "max" or int(raw) >= UNLIMITED:
                self.no_limit = f"{self.limit_file} is {raw}, no limit"
            else:
                self.limit = int(raw)
        except (OSError, ValueError) as e:
            self.no_limit = f"{self.limit_file} could not be read ({e})"

    def record(self):
        return {"t": self.t, "current": self.current, "max": self.limit, **self.counters}


def files_under(paths):
    """(path, stat or None when it cannot be read) for every regular file under the paths, once each. Directory links
    are followed, so a hub cache kept on another disk behind a link is reached; a directory already walked is not
    walked again, so a link back up the tree cannot loop."""
    seen_files, seen_dirs = set(), set()

    def walk(top):
        if os.path.isfile(top):
            yield top
            return
        for parent, dirs, names in os.walk(top, followlinks=True):
            try:
                here = os.stat(parent)
            except OSError:
                continue
            if (here.st_dev, here.st_ino) in seen_dirs:
                dirs[:] = []
                continue
            seen_dirs.add((here.st_dev, here.st_ino))
            yield from (os.path.join(parent, name) for name in names)

    for top in paths:
        for path in walk(top):
            try:
                # stat follows symlinks, so the HF cache's snapshots/ links and the blobs/ files they point at count
                # once between them.
                info = os.stat(path)
            except OSError:
                yield path, None
                continue
            if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) not in seen_files:
                seen_files.add((info.st_dev, info.st_ino))
                yield path, info


def drop_cache(paths, sync=False):
    """Asks the kernel to drop every regular file under the paths from the page cache. Returns (files, bytes,
    failures); pages a live process has mapped stay. `sync` writes dirty pages back first, which fadvise skips: what a
    download just wrote is still dirty."""
    advise = getattr(os, "posix_fadvise", None)
    files = size = failed = 0
    for path, info in files_under(paths):
        if info is None:
            failed += 1
            continue
        if advise:
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    if sync:
                        os.fdatasync(fd)
                    advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                failed += 1
                continue
        files += 1
        size += info.st_size
    return files, size, failed


def read_stat(path):
    """ppid, state, start time, RSS pages, and command name from a /proc stat file, or None when it is gone. stat never
    takes the process's mmap lock, which cmdline does and which a worker stuck in a page fault holds."""
    try:
        text = read_text(path)
    except OSError:
        return None
    close = text.rfind(")")
    fields = text[close + 2:].split()
    try:
        return Stat(int(fields[1]), fields[0], int(fields[19]), int(fields[21]), text[text.find("(") + 1:close])
    except (IndexError, ValueError):
        return None


def all_stats():
    try:
        pids = [int(name) for name in os.listdir(PROC) if name.isdigit()]
    except OSError:
        return {}
    stats = {}
    for pid in pids:
        found = read_stat(os.path.join(PROC, str(pid), "stat"))
        if found:
            stats[pid] = found
    return stats


class Server:
    """The server's process tree: the pid serve_and_smoke.sh writes and its descendants, found through /proc/*/stat
    parent ids. Pids stay known across polls, so a worker whose parent died is still killed; the start time recorded
    with each tells a live one from a reused pid."""

    def __init__(self, pid_file):
        self.pid_file = pid_file
        self.root = None
        self.known = {}

    def same(self, pid, stats):
        start = self.known[pid]
        if start is None:
            # Without /proc (a dev box) there is no start time to compare, and the pid is taken as is.
            return True
        found = stats.get(pid)
        return found is not None and found.state != "Z" and found.start == start

    def refresh(self):
        stats = all_stats()
        if self.root is None:
            try:
                self.root = int(read_text(self.pid_file).strip())
            except (OSError, ValueError):
                return
            # A server that already exited is not tracked: its pid may belong to another process by now.
            if stats and self.root not in stats:
                return
            self.known[self.root] = stats[self.root].start if self.root in stats else None
        children = collections.defaultdict(list)
        for pid, found in stats.items():
            children[found.ppid].append(pid)
        queue = [pid for pid in self.known if self.same(pid, stats)]
        while queue:
            for child in children[queue.pop()]:
                if child not in self.known:
                    self.known[child] = stats[child].start
                    queue.append(child)

    def live(self):
        stats = all_stats()
        pids = []
        for pid in self.known:
            if not self.same(pid, stats):
                continue
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            pids.append(pid)
        return pids

    def signal(self, signum):
        sent = []
        for pid in self.live():
            try:
                os.kill(pid, signum)
                sent.append(pid)
            except OSError as e:
                log(f"signal {signum} to {pid} failed: {e}")
        return sent


def start(argv):
    # Output goes to a file rather than a pipe, so reading it back can never block on a child that hangs.
    out = tempfile.TemporaryFile()
    try:
        return argv, out, subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    except OSError as e:
        return argv, out, e


def finish(started, deadline):
    """The command line, how it ended, and its output."""
    argv, out, proc = started
    with out:
        if isinstance(proc, OSError):
            return f"$ {shlex.join(argv)}\n{proc}\n"
        try:
            ended = f"exit {proc.wait(timeout=max(0.0, deadline - time.monotonic()))}"
        except subprocess.TimeoutExpired:
            # A child stuck in an uninterruptible read cannot be reaped until the read returns, so it is not waited
            # for: the kill after it matters more.
            proc.kill()
            ended = "killed at its time limit"
        out.seek(0)
        return f"$ {shlex.join(argv)} ({ended})\n{out.read().decode('utf-8', 'replace')}"


def process_table():
    stats = all_stats()
    if not stats:
        return f"{PROC} lists no processes\n"
    page = os.sysconf("SC_PAGE_SIZE")
    rows = sorted(stats.items(), key=lambda item: -item[1].rss)[:TABLE_LINES]
    lines = ["    PID    PPID S    RSS_MIB COMM"]
    lines += [f"{pid:7} {s.ppid:7} {s.state} {s.rss * page / 2**20:10.0f} {s.comm}" for pid, s in rows]
    return "\n".join(lines) + "\n"


def describe(pid):
    """A stopped server process: its memory lines, each thread's state and wait channel, and the kernel stack of each
    thread in uninterruptible sleep, where the container may read it."""
    base = os.path.join(PROC, str(pid))
    out = [f"\n## pid {pid}\n"]
    try:
        out.append("".join(line + "\n" for line in read_text(os.path.join(base, "status")).splitlines()
                           if line.split(":", 1)[0] in STATUS_KEYS))
    except OSError as e:
        out.append(f"status: {e}\n")
    try:
        tids = sorted(os.listdir(os.path.join(base, "task")), key=lambda tid: int(tid) if tid.isdigit() else 0)
    except OSError as e:
        out.append(f"task: {e}\n")
        return "".join(out)
    for tid in tids:
        task = os.path.join(base, "task", tid)
        found = read_stat(os.path.join(task, "stat"))
        try:
            wchan = read_text(os.path.join(task, "wchan")).strip() or "-"
        except OSError as e:
            wchan = f"({e.strerror})"
        out.append(f"task {tid} {found.state if found else '?'} {found.comm if found else '?'} wchan {wchan}\n")
        if found and found.state == "D":
            try:
                out.append(read_text(os.path.join(task, "stack")))
            except OSError as e:
                out.append(f"  stack: {e}\n")
    return "".join(out)


def stacks(pids):
    """Python stacks of the stopped server, all dumped at once within one budget."""
    if shutil.which("py-spy") is None:
        return "py-spy is not on PATH\n"
    # --nonblocking: the default dump stops the process first, and a worker in uninterruptible sleep never stops.
    started = [start(["py-spy", "dump", "--nonblocking", "--pid", str(pid)]) for pid in pids]
    deadline = time.monotonic() + STACK_BUDGET_S
    return "".join(finish(one, deadline) for one in started)


class Guard:
    def __init__(self, run_dir, parent, pattern, cache_paths):
        self.run_dir = run_dir
        self.parent = parent
        self.pattern = pattern
        self.cache_paths = cache_paths
        self.root = cgroup_root()
        self.server = Server(os.path.join(run_dir, "server.pid"))
        self.poll_s = knob("MEM_GUARD_POLL_S", 5.0)
        self.kill_at = knob("MEM_GUARD_KILL_FRACTION", 0.93)
        # The kill line is checked after the trim line's trim, so the trim line cannot sit above it.
        self.trim_at = min(knob("MEM_GUARD_TRIM_FRACTION", 0.80), self.kill_at)
        self.last_trim = None
        self.last_freed = None
        self.warned = False
        self.fired = False

    def path(self, name):
        return os.path.join(self.run_dir, name)

    def watch(self):
        try:
            # Whether a container may ask for reclaim, or set a soft limit, differs between hosts and decides what a
            # guard can do there beyond fadvise.
            directory = cgroup_files(self.root)[0]
            append(self.path("memory.jsonl"),
                   {"t": utc_now(), "controls": {name: control(directory, name) for name in CONTROLS}})
        except OSError as e:
            log(f"recording the cgroup controls failed: {e}")
        # serve_variant.sh stops the guard on every way out it controls. Killed outright it cannot, and a guard left
        # behind would watch the next variant's run with this one's state.
        while os.getppid() == self.parent:
            try:
                self.poll()
            # A reading that fails now (a file briefly missing, a full disk) may work at the next poll, and a guard
            # that exited would leave the rest of the run unwatched.
            except Exception as e:
                log(f"poll failed: {type(e).__name__}: {e}")
            time.sleep(self.poll_s)

    def poll(self):
        reading = Reading(self.root)
        self.server.refresh()
        record = reading.record()
        if reading.limit is None:
            if not self.warned:
                self.warned = True
                append(self.path("memory.jsonl"),
                       {"t": reading.t, "warn": f"{reading.no_limit}, so the guard only logs: no trim, no kill"})
            append(self.path("memory.jsonl"), record)
            return
        if self.fired:
            append(self.path("memory.jsonl"), record)
            return
        kill_line = self.kill_at * reading.limit
        level = reading.current
        if reading.current >= self.trim_at * reading.limit and self.trim_due(reading.current >= kill_line):
            record["trim"], after = self.trim(reading, kill_line)
            record["after_trim"] = after
            if after is not None:
                level = after
        # Before serve_and_smoke.sh writes the server's pid there is no server of this run to stop, and the guard fires
        # only once.
        fire = level >= kill_line and os.path.exists(self.server.pid_file)
        if fire:
            record["fired"] = True
        append(self.path("memory.jsonl"), record)
        if fire:
            self.fire(reading, level)

    def trim_due(self, over_kill_line):
        if self.last_trim is not None and time.monotonic() - self.last_trim < TRIM_EVERY_S:
            return False
        # A useless last trim stops trims, except right before a kill, which costs a whole run.
        return over_kill_line or self.last_freed is None or self.last_freed >= USEFUL_TRIM

    def trim(self, reading, kill_line):
        drop_cache(self.cache_paths)
        if reading.current >= kill_line:
            # Over the kill line the server is stopped next; a reclaim of a cgroup growing this fast can take longer
            # than the headroom left.
            reclaim = "skipped"
        else:
            reclaim = self.reclaim(reading.directory, min(reading.current - int(self.trim_at * reading.limit),
                                                          RECLAIM_CAP))
        self.last_trim = time.monotonic()
        try:
            after = current_usage(self.root)
        except (OSError, ValueError) as e:
            log(f"reading the usage after a trim failed: {e}")
            after = None
        self.last_freed = reading.current - after if after is not None else 0
        return {"fadvise": hasattr(os, "posix_fadvise"), "reclaim": reclaim}, after

    def reclaim(self, directory, excess):
        """Asks the kernel to reclaim `excess` bytes of the cgroup. fadvise skips page cache a process has mapped, and
        the kernel reclaims that on its own only at memory.max, past the 98% where RunPod stops the container."""
        if control(directory, "memory.reclaim") != "writable":
            return "unavailable"
        try:
            with open(os.path.join(directory, "memory.reclaim"), "w", encoding="utf-8") as f:
                f.write(str(excess))
            return "ok"
        # EAGAIN means the kernel freed less than asked, which still counts: after_trim shows how much.
        except OSError as e:
            return errno.errorcode.get(e.errno, str(e))

    def fire(self, reading, level):
        self.fired = True
        self.server.refresh()
        # Stopped first: the memory stops growing, and the processes keep their state for the report.
        stopped = self.server.signal(signal.SIGSTOP)
        summary = (f"fired {reading.t}: memory.current {level / GIB:.1f} GiB is {level / reading.limit:.1%} of "
                   f"memory.max {reading.limit / GIB:.1f} GiB, at or over the kill line of {self.kill_at:.0%} after a "
                   f"page cache trim")
        log(f"{summary}; stopped {stopped}")
        report = self.path("memory_guard.txt")
        try:
            # progress.sh's line, written here so `harness exec` shows the guard firing while the report is written.
            append(self.path("progress.jsonl"),
                   {"phase": "mem_guard", "t": int(time.time()), "detail": f"{summary}; stopped pids {stopped}"})
            with open(report, "w", encoding="utf-8") as f:
                f.write(f"{summary}\n\n## reading\n{json.dumps(reading.record())}\n\n## memory.stat\n"
                        f"{reading.stat_text}\n## largest processes\n{process_table()}\n## server processes, stopped: "
                        f"{stopped}\n")
                for pid in stopped:
                    f.write(describe(pid))
                # What is written so far survives a py-spy that hangs past the container's end.
                f.flush()
                f.write(f"\n## stacks\n{stacks(stopped)}")
        except Exception as e:
            log(f"writing the report failed: {type(e).__name__}: {e}")
        # The kill is what keeps RunPod from removing the container, so a report that failed does not stop it.
        killed = self.server.signal(signal.SIGKILL)
        outcome = f"SIGKILL sent to {killed}\n"
        if self.server.root is None:
            # No server pid was ever written, so matching command lines is all that is left, wrappers whose arguments
            # name the server included.
            outcome += finish(start(["pkill", "-9", "-f", self.pattern]), time.monotonic() + SWEEP_TIMEOUT_S)
        log(outcome.strip())
        try:
            with open(report, "a", encoding="utf-8") as f:
                f.write(f"\n## kill\n{outcome}")
        except OSError as e:
            log(f"recording the kill failed: {e}")
        self.after_kill()

    def after_kill(self):
        # The server's mappings went with it, so the cache a trim could not drop while it ran drops now.
        deadline = time.monotonic() + EXIT_WAIT_S
        while self.server.live() and time.monotonic() < deadline:
            time.sleep(0.1)
        left = self.server.live()
        files, size, failed = drop_cache(self.cache_paths)
        try:
            after = current_usage(self.root)
        except (OSError, ValueError) as e:
            log(f"reading the usage after the kill failed: {e}")
            after = None
        append(self.path("memory.jsonl"), {"t": utc_now(), "after_kill": {"live": left, "dropped_files": files,
                                                                          "failed": failed, "current": after}})


def report_drop(paths):
    root = cgroup_root()

    def usage():
        try:
            return f"{current_usage(root) / GIB:.1f} GiB"
        except (OSError, ValueError):
            return "unreadable"

    before = usage()
    files, size, failed = drop_cache(paths, sync=True)
    done = "dropped" if hasattr(os, "posix_fadvise") else "found (no posix_fadvise here, so none dropped)"
    print(f"drop-cache: {done} {files} files, {size / 1e9:.1f} GB, {failed} failed; memory.current {before} -> "
          f"{usage()}", flush=True)


def report_limit():
    try:
        reading = Reading(cgroup_root())
    except (OSError, ValueError) as e:
        print(f"none the cgroup could not be read ({e})")
        return
    print(f"{reading.limit} {reading.limit_file}" if reading.limit is not None else f"none {reading.no_limit}")


def main(args):
    if args[:1] == ["--drop-cache"]:
        report_drop(args[1:])
        return 0
    if args == ["--limit"]:
        report_limit()
        return 0
    if len(args) < 3 or not args[1].isdigit():
        print(__doc__, file=sys.stderr)
        return 2
    Guard(args[0], int(args[1]), args[2], args[3:]).watch()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
