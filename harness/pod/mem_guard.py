"""Watches the container's memory while a variant runs, so a server that fills it is ended by the harness while the
container and its logs still exist. RunPod stops a container at about 98% of its memory limit and removes it, and
every file on it goes too; vllm#56771 attempt 6 lost its server.log, and any trace of what filled the memory, that way.

The first line of <run_dir>/memory.jsonl, controls, says whether the cgroup's memory.reclaim and memory.high exist and
are writable here. Then every MEM_GUARD_POLL_S seconds (default 15) one line: t, current, max (null for no limit), and
the memory.stat counters that tell anonymous memory, page cache, shmem, and locked pages apart. At
MEM_GUARD_TRIM_FRACTION of the limit (default 0.80), at most once a minute, the guard trims: it drops the cache paths
from the page cache, then writes the bytes over the trim line to memory.reclaim when it can, and the line carries trim
(fadvise: whether that pass could run, reclaim: ok, unavailable, or the errno name) and the usage after, after_trim.
memory.high and memory.max are never written. At MEM_GUARD_KILL_FRACTION (default 0.93) on a poll after a trim the
guard adds a mem_guard phase to <run_dir>/progress.jsonl, writes <run_dir>/memory_guard.txt (its reading on the first
line, then memory.stat, the largest processes, and each server process's memory lines and Python stack), kills every
process whose command line matches <server_pattern> as `pkill -f` does, and exits. Under no limit, or one it cannot
read, it only logs.
--drop-cache drops every regular file under the paths from the page cache once and prints one summary line.

Usage: mem_guard.py <run_dir> <server_pattern> [<cache_path>...]
       mem_guard.py --drop-cache <path>...
  MEM_GUARD_CGROUP (default /sys/fs/cgroup) is the cgroup v2 directory, or the v1 root that holds memory/.
"""

import datetime
import errno
import json
import os
import shlex
import shutil
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
TRIM_EVERY_S = 60
COMMAND_TIMEOUT_S = 60
PS_LINES = 40
STATUS_KEYS = ("VmRSS", "VmLck", "VmPin", "RssAnon", "RssFile", "RssShmem")
CONTROLS = ("memory.reclaim", "memory.high")
GIB = 2**30


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message):
    print(f"{utc_now()} {message}", file=sys.stderr, flush=True)


def append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def knob(name, default):
    value = os.environ.get(name, "")
    try:
        return float(value) if value else default
    except ValueError:
        log(f"{name}={value!r} is not a number; using {default}")
        return default


def cgroup_files(root):
    """The usage, limit, and stat files, and the memory.stat names to try for each counter: v2 when present, else v1."""
    if os.path.exists(os.path.join(root, "memory.current")):
        files = [os.path.join(root, name) for name in ("memory.current", "memory.max", "memory.stat")]
        return files, {key: [key] for key in COUNTERS}
    v1 = os.path.join(root, "memory")
    files = [os.path.join(v1, name) for name in ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.stat")]
    return files, {key: ["total_" + V1_NAMES.get(key, key), V1_NAMES.get(key, key)] for key in COUNTERS}


def current_usage(root):
    return int(read_text(cgroup_files(root)[0][0]))


def control(root, name):
    """missing, read-only (container runtimes often mount the cgroup tree so), or writable."""
    path = os.path.join(root, name)
    if not os.path.exists(path):
        return "missing"
    return "writable" if os.access(path, os.W_OK) else "read-only"


class Reading:
    """One look at the cgroup: usage, limit (None for none, with no_limit saying why), the logged counters, and
    memory.stat as the kernel wrote it."""

    def __init__(self, root):
        (_, limit_file, stat_file), names = cgroup_files(root)
        self.t = utc_now()
        self.current = current_usage(root)
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
            raw = read_text(limit_file).strip()
            if raw == "max" or int(raw) >= UNLIMITED:
                self.no_limit = f"{limit_file} is {raw}, no limit"
            else:
                self.limit = int(raw)
        except (OSError, ValueError) as e:
            self.no_limit = f"{limit_file} could not be read ({e})"

    def record(self):
        return {"t": self.t, "current": self.current, "max": self.limit, **self.counters}


def drop_cache(paths):
    """Asks the kernel to drop every regular file under the paths from the page cache. Returns (files, bytes,
    failures); pages a live process has mapped stay."""
    advise = getattr(os, "posix_fadvise", None)
    seen = set()
    files = size = failed = 0
    for top in paths:
        for parent, _, names in os.walk(top):
            for name in names:
                path = os.path.join(parent, name)
                try:
                    # stat follows symlinks, so the HF cache's snapshots/ links and the blobs/ files they point at
                    # count once between them.
                    info = os.stat(path)
                    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) in seen:
                        continue
                    seen.add((info.st_dev, info.st_ino))
                    if advise:
                        fd = os.open(path, os.O_RDONLY)
                        try:
                            advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        finally:
                            os.close(fd)
                except OSError:
                    failed += 1
                    continue
                files += 1
                size += info.st_size
    return files, size, failed


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
            ended = f"killed after {COMMAND_TIMEOUT_S} s"
        out.seek(0)
        return f"$ {shlex.join(argv)} ({ended})\n{out.read().decode('utf-8', 'replace')}"


def run(argv):
    return finish(start(argv), time.monotonic() + COMMAND_TIMEOUT_S)


def memory_lines(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            return "".join(line for line in f if line.split(":", 1)[0] in STATUS_KEYS)
    except OSError as e:
        return f"{e}\n"


def stacks(pids):
    """Python stacks of the server processes, all dumped at once, so stuck workers cost one timeout between them."""
    if shutil.which("py-spy") is None:
        return "py-spy is not on PATH\n"
    # The default dump stops the process first, and a worker in uninterruptible sleep cannot stop, so the dump would
    # wait out its whole timeout.
    started = [start(["py-spy", "dump", "--nonblocking", "--pid", str(pid)]) for pid in pids]
    deadline = time.monotonic() + COMMAND_TIMEOUT_S
    return "".join(finish(one, deadline) for one in started)


class Guard:
    def __init__(self, run_dir, pattern, cache_paths):
        self.run_dir = run_dir
        self.pattern = pattern
        self.cache_paths = cache_paths
        self.root = os.environ.get("MEM_GUARD_CGROUP", "/sys/fs/cgroup")
        self.poll_s = knob("MEM_GUARD_POLL_S", 15.0)
        self.kill_at = knob("MEM_GUARD_KILL_FRACTION", 0.93)
        # A kill only ever follows a trim attempt, so the trim line cannot sit above the kill line.
        self.trim_at = min(knob("MEM_GUARD_TRIM_FRACTION", 0.80), self.kill_at)
        self.last_trim = None
        self.warned = False

    def path(self, name):
        return os.path.join(self.run_dir, name)

    def watch(self):
        try:
            # Whether a container may ask for reclaim, or set a soft limit, differs between hosts and decides what a
            # guard can do there beyond fadvise.
            append(self.path("memory.jsonl"),
                   {"t": utc_now(), "controls": {name: control(self.root, name) for name in CONTROLS}})
        except OSError as e:
            log(f"recording the cgroup controls failed: {e}")
        parent = os.getppid()
        # serve_variant.sh stops the guard on every way out it controls. Killed outright it cannot, and a guard left
        # behind would kill the next variant's server, so the guard ends with its parent.
        while os.getppid() == parent:
            try:
                if self.poll():
                    return
            # A reading that fails now (a file briefly missing, a full disk) may work at the next poll, and a guard
            # that exited would leave the rest of the run unwatched.
            except Exception as e:
                log(f"poll failed: {type(e).__name__}: {e}")
            time.sleep(self.poll_s)

    def poll(self):
        """Takes and logs one reading; True once the guard has fired."""
        reading = Reading(self.root)
        record = reading.record()
        if reading.limit is None:
            if not self.warned:
                self.warned = True
                append(self.path("memory.jsonl"),
                       {"t": reading.t, "warn": f"{reading.no_limit}, so the guard only logs: no trim, no kill"})
            append(self.path("memory.jsonl"), record)
            return False
        now = time.monotonic()
        trim_due = self.last_trim is None or now - self.last_trim >= TRIM_EVERY_S
        trimmed = False
        if reading.current >= self.trim_at * reading.limit and trim_due:
            drop_cache(self.cache_paths)
            self.last_trim, trimmed = now, True
            record["trim"] = {"fadvise": hasattr(os, "posix_fadvise"),
                              "reclaim": self.reclaim(reading.current - int(self.trim_at * reading.limit))}
            try:
                record["after_trim"] = current_usage(self.root)
            except (OSError, ValueError) as e:
                log(f"reading the usage after a trim failed: {e}")
                record["after_trim"] = None
        # Page cache alone can fill the cgroup while no process needs the memory, so a kill waits until a trim has
        # had a poll to show what it freed.
        fired = reading.current >= self.kill_at * reading.limit and self.last_trim is not None and not trimmed
        if fired:
            record["fired"] = True
        append(self.path("memory.jsonl"), record)
        if fired:
            self.fire(reading)
        return fired

    def reclaim(self, excess):
        """Asks the kernel to reclaim `excess` bytes of the cgroup. fadvise skips page cache a process has mapped, as
        vLLM maps the checkpoint while it copies the weights, and the kernel reclaims that on its own only at
        memory.max, past the 98% where RunPod stops the container."""
        if control(self.root, "memory.reclaim") != "writable":
            return "unavailable"
        try:
            with open(os.path.join(self.root, "memory.reclaim"), "w", encoding="utf-8") as f:
                f.write(str(excess))
            return "ok"
        # EAGAIN means the kernel freed less than asked, which still counts: the next poll shows how much.
        except OSError as e:
            return errno.errorcode.get(e.errno, str(e))

    def fire(self, reading):
        summary = (f"fired {reading.t}: memory.current {reading.current / GIB:.1f} GiB is "
                   f"{reading.current / reading.limit:.1%} of memory.max {reading.limit / GIB:.1f} GiB, at or over the "
                   f"kill line of {self.kill_at:.0%} after a page cache trim")
        log(summary)
        report = self.path("memory_guard.txt")
        try:
            # progress.sh's line, written here so `harness exec` shows the guard firing while the report is written.
            append(self.path("progress.jsonl"), {"phase": "mem_guard", "t": int(time.time()), "detail": summary})
            with open(report, "w", encoding="utf-8") as f:
                f.write(f"{summary}\n\n## reading\n{json.dumps(reading.record())}\n\n## memory.stat\n"
                        f"{reading.stat_text}\n## largest processes\n")
                ps = run(["ps", "-eo", "pid,ppid,stat,rss,pcpu,etime,args", "--sort=-rss"])
                f.write("\n".join(ps.splitlines()[:PS_LINES + 1]) + "\n\n## server processes\n")
                listing = run(["pgrep", "-f", self.pattern])
                f.write(listing)
                pids = [int(line) for line in listing.splitlines()[1:] if line.strip().isdigit()]
                for pid in pids:
                    f.write(f"\n## /proc/{pid}/status\n{memory_lines(pid)}")
                # What is written so far survives a py-spy that hangs past the container's end.
                f.flush()
                f.write(f"\n## stacks\n{stacks(pids)}")
        except Exception as e:
            log(f"writing the report failed: {type(e).__name__}: {e}")
        # The kill is what keeps RunPod from removing the container, so a report that failed does not stop it.
        outcome = run(["pkill", "-9", "-f", self.pattern])
        log(outcome.strip())
        try:
            with open(report, "a", encoding="utf-8") as f:
                f.write(f"\n## kill\n{outcome}")
        except OSError as e:
            log(f"recording the kill failed: {e}")


def report_drop(paths):
    root = os.environ.get("MEM_GUARD_CGROUP", "/sys/fs/cgroup")

    def usage():
        try:
            return f"{current_usage(root) / GIB:.1f} GiB"
        except (OSError, ValueError):
            return "unreadable"

    before = usage()
    files, size, failed = drop_cache(paths)
    done = "dropped" if hasattr(os, "posix_fadvise") else "found (no posix_fadvise here, so none dropped)"
    print(f"drop-cache: {done} {files} files, {size / 1e9:.1f} GB, {failed} failed; memory.current {before} -> "
          f"{usage()}", flush=True)


def main(args):
    if args[:1] == ["--drop-cache"]:
        report_drop(args[1:])
        return 0
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    Guard(args[0], args[1], args[2:]).watch()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
