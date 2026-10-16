#!/usr/bin/env bash
# Runs mem_guard.py against fake cgroup and /proc directories with a 0.1 s poll: logging under the trim line; a trim
# over it with a capped memory.reclaim request; the fire over the kill line, which stops, reports on, and kills the
# server's process tree (stand-in processes) and nothing else; a SIGTERM mid-fire, which still kills and records; no
# fire while server.pid holds no pid; a limit of max; cgroup v1 names; a container cgroup found through
# /proc/self/cgroup, and --limit; a cgroup it cannot read; a guard whose parent dies; --drop-cache through a hub kept
# behind a link. Then the trim and kill decisions, driven poll by poll, and a kill that survives a failing report and
# log. A fake py-spy records its calls.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
guard="$here/../../pod/mem_guard.py"
work=$(mktemp -d)
pids=()
cleanup() {
    local pid
    for pid in "${pids[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    rm -rf "$work"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    local log
    for log in "$work"/*/mem_guard.log; do
        echo "--- $log" >&2
        cat "$log" >&2
    done
    exit 1
}

mkdir -p "$work/bin"
cat >"$work/bin/py-spy" <<'EOF'
#!/usr/bin/env bash
echo "py-spy $*"
EOF
chmod +x "$work/bin/py-spy"
export PATH="$work/bin:$PATH"

# Writes a cgroup v2 directory. Args: dir current max
cgroup_v2() {
    mkdir -p "$1"
    echo "$2" >"$1/memory.current"
    echo "$3" >"$1/memory.max"
    printf '%s\n' 'anon 11' 'file 22' 'kernel 99' 'active_file 33' 'inactive_file 44' 'shmem 55' 'file_mapped 66' \
        'file_dirty 77' 'unevictable 88' >"$1/memory.stat"
}

# Starts a stand-in process whose command line is $1 and prints its pid. A subshell starts it, so it is no job of
# this shell's and bash does not report its kill.
spawn() {
    (bash -c 'exec -a "$0" sleep 300' "$1" >/dev/null 2>&1 & echo $!)
}

# Writes a fake /proc entry: stat, status, and one thread with a wait channel, plus a kernel stack in state D.
# Args: proc_dir pid ppid state start_time
fake_proc() {
    local dir=$1/$2
    mkdir -p "$dir/task/$2"
    printf '%s (sleep) %s %s %s%s 0 2048\n' "$2" "$4" "$3" "$(printf '0 %.0s' $(seq 17))" "$5" >"$dir/stat"
    cp "$dir/stat" "$dir/task/$2/stat"
    printf 'Name:\tsleep\nState:\t%s\nVmRSS:\t    8192 kB\nVmLck:\t       0 kB\nVmPin:\t       0 kB\nThreads:\t1\n' \
        "$4" >"$dir/status"
    echo folio_wait_bit_common >"$dir/task/$2/wchan"
    if [[ $4 == D ]]; then
        echo '[<0>] fuse_wait_answer+0x5d/0xa0' >"$dir/task/$2/stack"
    fi
}

# Starts the guard for case $1 on $work/$1/cgroup with its run dir $work/$1/run; its pid lands in guard_pid.
# Args: case [cache_path...]
start_guard() {
    local dir=$work/$1
    shift
    mkdir -p "$dir/run"
    MEM_GUARD_CGROUP=$dir/cgroup MEM_GUARD_POLL_S=0.1 python3 -I "$guard" "$dir/run" "$$" "$@" \
        >"$dir/mem_guard.log" 2>&1 &
    guard_pid=$!
    pids+=("$guard_pid")
}

# Waits up to 10 s for a command to succeed.
wait_for() {
    for _ in $(seq 100); do
        if "$@"; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

gone() {
    ! kill -0 "$1" 2>/dev/null
}

has_lines() {
    [[ -f $1 && $(wc -l <"$1") -ge $2 ]]
}

# Ends a guard that should still be running.
stop() {
    kill -0 "$guard_pid" 2>/dev/null || fail "$1: the guard exited on its own"
    kill "$guard_pid"
    wait "$guard_pid" || true
}

# Checks a case's memory.jsonl: a Python expression over its parsed lines, `rows`, must hold. Args: case expression
rows() {
    python3 -I - "$work/$1/run/memory.jsonl" "$2" <<'EOF' || fail "$1: $2"
import json
import sys

path, check = sys.argv[1:]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
if not eval(f"({check})", {"rows": rows}):
    sys.exit(f"{check} does not hold for {rows}")
EOF
}

missing_controls="{'controls': {'memory.reclaim': 'missing', 'memory.high': 'missing'}, 't': rows[0]['t']}"
fadvise=$(python3 -I -c 'import os; print(hasattr(os, "posix_fadvise"))')

cgroup_v2 "$work/low/cgroup" 100000 1000000
start_guard low
wait_for has_lines "$work/low/run/memory.jsonl" 4 || fail "low: fewer than 4 lines in 10 s"
stop low
rows low "rows[0] == $missing_controls and all(r['current'] == 100000 and r['max'] == 1000000 and r['anon'] == 11
          and r['file'] == 22 and r['active_file'] == 33 and r['inactive_file'] == 44 and r['shmem'] == 55
          and r['file_mapped'] == 66 and r['file_dirty'] == 77 and r['unevictable'] == 88 and r['t'].endswith('Z')
          and not {'trim', 'fired', 'warn'} & r.keys() for r in rows[1:])"
echo "ok: under the trim line every poll logs usage, limit, and the memory.stat counters"

# 85 GiB of a 100 GiB limit: 5 GiB over the trim line, of which one request asks for 4.
cgroup_v2 "$work/trim/cgroup" 91268055040 107374182400
: >"$work/trim/cgroup/memory.reclaim"
start_guard trim "$work/trim/cache"
wait_for has_lines "$work/trim/run/memory.jsonl" 6 || fail "trim: fewer than 6 lines in 10 s"
stop trim
rows trim "rows[0]['controls'] == {'memory.reclaim': 'writable', 'memory.high': 'missing'}
           and rows[1]['trim'] == {'fadvise': $fadvise, 'reclaim': 'ok'} and rows[1]['after_trim'] == 91268055040
           and not any('trim' in r or 'fired' in r for r in rows[2:])"
[[ $(<"$work/trim/cgroup/memory.reclaim") == 4294967296 ]] \
    || fail "trim: memory.reclaim holds $(<"$work/trim/cgroup/memory.reclaim"), not the 4 GiB cap"
[[ ! -e $work/trim/run/memory_guard.txt ]] || fail "trim: the guard fired under the kill line"
echo "ok: over the trim line the guard trims once, asks memory.reclaim for at most 4 GiB, and logs what is left"

# The server, its worker, and a process outside its tree.
server=$(spawn "fakeserver-$$")
worker=$(spawn "fakeserver-$$-worker")
outsider=$(spawn "fakeserver-$$-outsider")
pids+=("$server" "$worker" "$outsider")
proc=$work/full/proc
fake_proc "$proc" "$server" 1 D 500
fake_proc "$proc" "$worker" "$server" S 501
fake_proc "$proc" "$outsider" 1 S 502
cgroup_v2 "$work/full/cgroup" 950000 1000000
mkdir -p "$work/full/run"
echo "$server" >"$work/full/run/server.pid"
MEM_GUARD_PROC=$proc start_guard full "$work/full/cache"
report=$work/full/run/memory_guard.txt
wait_for grep -qs '"after_kill"' "$work/full/run/memory.jsonl" || fail "full: the guard did not fire"
wait_for has_lines "$work/full/run/memory.jsonl" 6 || fail "full: the guard stopped logging after the kill"
stop full
for pid in "$server" "$worker"; do
    gone "$pid" || fail "full: $pid of the server's tree outlived the kill"
done
! gone "$outsider" || fail "full: the guard killed a process outside the server's tree"
rows full "rows[1]['trim'] == {'fadvise': $fadvise, 'reclaim': 'skipped'} and rows[1]['after_trim'] == 950000
           and rows[1]['fired'] is True and rows[2]['after_kill'] == {'live': [], 'current': 950000}
           and rows[3]['after_drop']['failed'] == 0 and not any('trim' in r or 'fired' in r for r in rows[4:])"
reading='^fired [0-9T:-]+Z: memory.current 0.0 GiB is 95.0% of memory.max 0.0 GiB, at or over the kill line of 93%'
head -n 1 "$report" | grep -qE "$reading after a page cache trim\$" \
    || fail "full: the report's first line is not the reading: $(head -n 1 "$report")"
grep -qx 'kernel 99' "$report" || fail "full: memory.stat is not in the report whole"
grep -qE "^ *$outsider +1 S +[0-9]+ sleep$" "$report" || fail "full: the process table has no row for every process"
grep -qx "## server processes, stopped: \[$server, $worker\]" "$report" || fail "full: the tree was not stopped"
grep -qxF "VmRSS:	    8192 kB" "$report" || fail "full: no memory lines"
grep -q '^Threads' "$report" && fail "full: status lines beyond the memory ones"
grep -qx "task $server D sleep wchan folio_wait_bit_common" "$report" || fail "full: no thread line"
grep -qxF '[<0>] fuse_wait_answer+0x5d/0xa0' "$report" || fail "full: no kernel stack of the thread in D"
grep -qx "py-spy dump --nonblocking --pid $worker" "$report" || fail "full: no Python stack of the worker"
grep -qx "SIGKILL sent to \[$server, $worker\]" "$report" || fail "full: the kill is not recorded"
grep -q pkill "$report" && fail "full: the guard swept by command line with a server pid known"
python3 -I - "$work/full/run/progress.jsonl" "$server" "$worker" <<'EOF' || fail "full: no mem_guard phase"
import json
import sys

path, server, worker = sys.argv[1:]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
assert [row["phase"] for row in rows] == ["mem_guard"], rows
assert rows[0]["detail"].startswith("fired ") and rows[0]["detail"].endswith(f"stopped pids [{server}, {worker}]"), rows
EOF
echo "ok: over the kill line the guard stops, reports on, and kills the server's tree only, then keeps logging"

# serve_variant.sh's SIGTERM lands while a py-spy that hangs holds the fire in its report. The kill and its record still
# have to land inside the 5 s serve_variant.sh gives the guard before SIGKILL.
mkdir -p "$work/hangbin"
printf '#!/usr/bin/env bash\nexec sleep 30\n' >"$work/hangbin/py-spy"
chmod +x "$work/hangbin/py-spy"
stuck=$(spawn "fakeserver-$$-stuck")
pids+=("$stuck")
cgroup_v2 "$work/term/cgroup" 950000 1000000
mkdir -p "$work/term/run"
echo "$stuck" >"$work/term/run/server.pid"
PATH="$work/hangbin:$PATH" start_guard term
wait_for grep -qs '^## server processes' "$work/term/run/memory_guard.txt" || fail "term: the guard did not fire"
sleep 0.3
started=$(date +%s)
kill "$guard_pid"
wait_for gone "$guard_pid" || fail "term: the guard outlived its SIGTERM by 10 s"
(( $(date +%s) - started <= 4 )) || fail "term: the guard took more than 4 s to finish after SIGTERM"
wait_for gone "$stuck" || fail "term: the server outlived the guard's SIGTERM"
rows term "rows[2]['after_kill']['live'] == [] and 'after_drop' in rows[3]"
grep -qx "SIGKILL sent to \[$stuck\]" "$work/term/run/memory_guard.txt" || fail "term: the kill is not recorded"
grep -q "^\$ py-spy dump --nonblocking --pid $stuck (killed before it finished)" "$work/term/run/memory_guard.txt" \
    || fail "term: the hung py-spy was not cut short"
echo "ok: a SIGTERM mid-fire cuts the report short, and the kill and its record still land"

cgroup_v2 "$work/nopid/cgroup" 950000 1000000
mkdir -p "$work/nopid/run"
echo garbled >"$work/nopid/run/server.pid"
start_guard nopid
wait_for has_lines "$work/nopid/run/memory.jsonl" 4 || fail "nopid: fewer than 4 lines in 10 s"
stop nopid
rows nopid "not any('fired' in r for r in rows)"
[[ ! -e $work/nopid/run/memory_guard.txt ]] || fail "nopid: the guard fired without a server pid"
echo "ok: while server.pid holds no pid the guard never fires"

cgroup_v2 "$work/max/cgroup" 990000 max
start_guard max "$work/max/cache"
wait_for has_lines "$work/max/run/memory.jsonl" 5 || fail "max: fewer than 5 lines in 10 s"
stop max
rows max "'is max, no limit, so the guard only logs' in rows[1]['warn'] and not any('warn' in r for r in rows[2:])
          and all(r['max'] is None and not {'trim', 'fired'} & r.keys() for r in rows[2:])"
[[ ! -e $work/max/run/memory_guard.txt ]] || fail "max: the guard fired without a limit"
echo "ok: with a limit of max the guard warns once and only logs"

mkdir -p "$work/v1/cgroup/memory"
echo 500 >"$work/v1/cgroup/memory/memory.usage_in_bytes"
echo 9223372036854771712 >"$work/v1/cgroup/memory/memory.limit_in_bytes"
printf '%s\n' 'cache 1' 'rss 2' 'mapped_file 3' 'dirty 4' 'shmem 5' 'total_cache 10' 'total_rss 20' \
    'total_mapped_file 30' 'total_dirty 40' 'total_shmem 50' 'total_active_file 60' 'total_inactive_file 70' \
    'total_unevictable 80' >"$work/v1/cgroup/memory/memory.stat"
start_guard v1
wait_for has_lines "$work/v1/run/memory.jsonl" 3 || fail "v1: fewer than 3 lines in 10 s"
stop v1
rows v1 "rows[0] == $missing_controls and 'warn' in rows[1] and rows[2] == {'t': rows[2]['t'], 'current': 500,
         'max': None, 'anon': 20, 'file': 10, 'active_file': 60, 'inactive_file': 70, 'shmem': 50, 'file_mapped': 30,
         'file_dirty': 40, 'unevictable': 80}"
echo "ok: cgroup v1 counters map to the v2 names, and its no-limit value means no limit"

# In the host's cgroup namespace the root is the host's, and the container's cgroup is the path /proc/self/cgroup names.
mkdir -p "$work/hostns/proc/self"
printf '0::/system.slice/docker-abc.scope\n' >"$work/hostns/proc/self/cgroup"
cgroup_v2 "$work/hostns/cgroup/system.slice/docker-abc.scope" 300000 1000000
MEM_GUARD_PROC=$work/hostns/proc start_guard hostns
wait_for has_lines "$work/hostns/run/memory.jsonl" 3 || fail "hostns: fewer than 3 lines in 10 s"
stop hostns
rows hostns "all(r['current'] == 300000 and r['max'] == 1000000 for r in rows[1:])"
limit=$(MEM_GUARD_PROC=$work/hostns/proc MEM_GUARD_CGROUP=$work/hostns/cgroup python3 -I "$guard" --limit)
[[ $limit == "1000000 $work/hostns/cgroup/system.slice/docker-abc.scope/memory.max" ]] || fail "hostns --limit: $limit"
limit=$(MEM_GUARD_CGROUP=$work/max/cgroup python3 -I "$guard" --limit)
[[ $limit == "none $work/max/cgroup/memory.max is max, no limit" ]] || fail "max --limit: $limit"
echo "ok: a container cgroup below the host's root is found through /proc/self/cgroup, by the guard and by --limit"

start_guard missing
wait_for grep -qs 'poll failed: FileNotFoundError' "$work/missing/mem_guard.log" || fail "missing: no error logged"
sleep 0.3
stop missing
echo "ok: a cgroup it cannot read is logged, and the guard keeps polling"

mkdir -p "$work/orphan/run"
cgroup_v2 "$work/orphan/cgroup" 100000 1000000
MEM_GUARD_CGROUP=$work/orphan/cgroup MEM_GUARD_POLL_S=0.1 bash -c 'python3 -I "$0" "$1" "$$" >"$2" 2>&1 &
echo $! >"$3"
sleep 2' "$guard" "$work/orphan/run" "$work/orphan/mem_guard.log" "$work/orphan/pid"
orphan=$(<"$work/orphan/pid")
pids+=("$orphan")
wait_for gone "$orphan" || fail "orphan: the guard outlived its parent"
has_lines "$work/orphan/run/memory.jsonl" 3 || fail "orphan: the guard did not run while its parent lived"
echo "ok: a guard whose parent dies without stopping it exits"

# HF_HOME/hub links to a hub on another disk, as issues/vllm-56771/setup.sh makes it, and a link inside it points back
# up the tree.
hub=$work/disk/hub/models--org--model
mkdir -p "$hub/blobs" "$hub/snapshots/abc" "$work/hf"
ln -s "$work/disk/hub" "$work/hf/hub"
printf '%01000d' 0 >"$hub/blobs/aaa"
printf '%0500d' 0 >"$hub/blobs/bbb"
echo outside >"$work/outside.txt"
ln -s ../../blobs/aaa "$hub/snapshots/abc/model.safetensors"
ln -s ../../blobs/bbb "$hub/snapshots/abc/config.json"
ln -s "$work/outside.txt" "$hub/snapshots/abc/outside.txt"
ln -s ../../blobs/missing "$hub/snapshots/abc/dangling"
ln -s ../.. "$hub/snapshots/abc/loop"
MEM_GUARD_CGROUP=$work/low/cgroup python3 -I "$guard" --drop-cache "$work/hf" "$work/nonexistent" >"$work/drop.out" \
    || fail "drop-cache exited non-zero"
done_re='(dropped|found \(no posix_fadvise here, so none dropped\))'
grep -qE "^drop-cache: $done_re 3 files, 0.0 GB, 1 failed; memory.current 0.0 GiB -> 0.0 GiB\$" \
    "$work/drop.out" || fail "drop-cache summary: $(cat "$work/drop.out")"
MEM_GUARD_CGROUP=$work/missing python3 -I "$guard" --drop-cache "$hub/blobs/aaa" >"$work/drop.out" \
    || fail "drop-cache exited non-zero on a file without a cgroup"
grep -qE "^drop-cache: $done_re 1 files, .*; memory.current unreadable -> unreadable\$" "$work/drop.out" \
    || fail "drop-cache on one file: $(cat "$work/drop.out")"
echo "ok: --drop-cache follows the hub link once, counts each file once, a dangling link as failed, and exits 0"

# Poll by poll, with the page cache pass and the clock under the test's control.
mkdir -p "$work/steps/run"
MEM_GUARD_PROC=$work/steps/noproc python3 -I - "$guard" "$work/steps" 2>"$work/steps/mem_guard.log" <<'EOF' \
    || fail "steps"
import errno
import importlib.util
import json
import os
import subprocess
import sys
import time

path, work = sys.argv[1:]
spec = importlib.util.spec_from_file_location("mem_guard", path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
cgroup = os.path.join(work, "cgroup")
os.makedirs(cgroup)
os.environ["MEM_GUARD_CGROUP"] = cgroup
run = os.path.join(work, "run")
clock = [1000.0]
time.monotonic = lambda: clock[0]


def usage(current):
    with open(os.path.join(cgroup, "memory.current"), "w") as f:
        f.write(str(current))


def last(key="current"):
    with open(os.path.join(run, "memory.jsonl")) as f:
        return [row for row in map(json.loads, f) if key in row][-1]


def spawn():
    # Orphaned at once, so its kill is reaped by init and not left a zombie of this process.
    out = subprocess.run(["bash", "-c", "sleep 300 >/dev/null 2>&1 & echo $!"], capture_output=True, text=True)
    return int(out.stdout)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class FullDisk:
    def write(self, text):
        raise OSError(errno.ENOSPC, "No space left on device")

    def flush(self):
        raise OSError(errno.ENOSPC, "No space left on device")


server = spawn()
with open(os.path.join(cgroup, "memory.max"), "w") as f:
    f.write("1000000")
with open(os.path.join(cgroup, "memory.stat"), "w") as f:
    f.write("anon 1\n")
with open(os.path.join(run, "server.pid"), "w") as f:
    f.write(str(server))
os.mkdir(os.path.join(cgroup, "memory.reclaim"))

# A trim that frees the usage back under the kill line decides that poll: no fire.
usage(950000)
guard.drop_cache = lambda paths, sync=False: (usage(100000), (0, 0, 0))[1]
step = guard.Guard(run, 0, [])
step.poll()
assert last()["trim"]["reclaim"] == "skipped" and last()["after_trim"] == 100000 and "fired" not in last(), last()

# Under the kill line a reclaim that fails records its errno; a trim that frees nothing stops the trims.
guard.drop_cache = lambda paths, sync=False: (0, 0, 0)
step = guard.Guard(run, 0, [])
usage(850000)
step.poll()
assert last()["trim"]["reclaim"] == "EISDIR", last()
clock[0] += 61
step.poll()
assert "trim" not in last(), last()

# Within a minute of a trim the kill line is decided on the reading alone. The report fails partway and the log is on
# a full disk, and the stopped server is still killed.
assert alive(server)
step = guard.Guard(run, 0, [])
step.poll()
assert "trim" in last(), last()
usage(950000)
clock[0] += 30


def broken(pid):
    raise RuntimeError("diagnostics broke")


guard.describe = broken
sys.stderr = FullDisk()
step.poll()
sys.stderr = sys.__stderr__
assert "trim" not in last() and last()["fired"] is True, last()
assert last("after_kill")["after_kill"]["live"] == [], last("after_kill")
assert not alive(server)
report = open(os.path.join(run, "memory_guard.txt")).read()
assert report.startswith("fired ") and f"## kill\nSIGKILL sent to [{server}]\n" in report, report

# Over the kill line a trim runs even after a useless one, and the fire waits for server.pid.
os.remove(os.path.join(run, "server.pid"))
step = guard.Guard(run, 0, [])
usage(850000)
clock[0] += 61
step.poll()
clock[0] += 61
usage(950000)
step.poll()
assert "trim" in last() and "fired" not in last(), last()
EOF
echo "ok: the kill is decided after that poll's trim, trims stop once useless, a fire needs a server pid, and the kill"
echo "    survives a report and a log that fail"
