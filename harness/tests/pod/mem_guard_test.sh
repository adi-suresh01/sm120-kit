#!/usr/bin/env bash
# Runs mem_guard.py against fake cgroup directories with a 0.1 s poll: logging under the trim line, one trim over it
# with a memory.reclaim request, the kill of a stand-in server over the kill line after a reclaim that failed, a limit
# of max, cgroup v1 names and its no-limit value, a cgroup it cannot read, a guard whose parent dies, and --drop-cache
# on a tree shaped like the HF cache. A fake py-spy records its calls.
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

# Starts the guard for case $1 on $work/$1/cgroup with its run dir $work/$1/run; its pid lands in guard_pid.
# Args: case pattern [cache_path...]
start_guard() {
    local dir=$work/$1
    shift
    mkdir -p "$dir/run"
    MEM_GUARD_CGROUP=$dir/cgroup MEM_GUARD_POLL_S=0.1 python3 -I "$guard" "$dir/run" "$@" >"$dir/mem_guard.log" 2>&1 &
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
start_guard low '[f]akeserver-none'
wait_for has_lines "$work/low/run/memory.jsonl" 4 || fail "low: fewer than 4 lines in 10 s"
stop low
rows low "rows[0] == $missing_controls and all(r['current'] == 100000 and r['max'] == 1000000 and r['anon'] == 11
          and r['file'] == 22 and r['active_file'] == 33 and r['inactive_file'] == 44 and r['shmem'] == 55
          and r['file_mapped'] == 66 and r['file_dirty'] == 77 and r['unevictable'] == 88 and r['t'].endswith('Z')
          and not {'trim', 'fired', 'warn'} & r.keys() for r in rows[1:])"
echo "ok: under the trim line every poll logs usage, limit, and the memory.stat counters"

cgroup_v2 "$work/trim/cgroup" 850000 1000000
: >"$work/trim/cgroup/memory.reclaim"
start_guard trim '[f]akeserver-none' "$work/trim/cache"
wait_for has_lines "$work/trim/run/memory.jsonl" 6 || fail "trim: fewer than 6 lines in 10 s"
stop trim
rows trim "rows[0]['controls'] == {'memory.reclaim': 'writable', 'memory.high': 'missing'}
           and rows[1]['trim'] == {'fadvise': $fadvise, 'reclaim': 'ok'} and rows[1]['after_trim'] == 850000
           and not any('trim' in r or 'fired' in r for r in rows[2:])"
[[ $(<"$work/trim/cgroup/memory.reclaim") == 50000 ]] \
    || fail "trim: memory.reclaim holds $(<"$work/trim/cgroup/memory.reclaim"), not the 50000 bytes over the line"
[[ ! -e $work/trim/run/memory_guard.txt ]] || fail "trim: the guard fired under the kill line"
echo "ok: over the trim line the guard trims once a minute, asks memory.reclaim for the excess, and logs what is left"

# Started from a subshell, so it is no job of this shell's, and bash does not report its kill on stderr.
(bash -c "exec -a fakeserver-$$ sleep 300" & echo $! >"$work/server.pid")
server=$(<"$work/server.pid")
pids+=("$server")
cgroup_v2 "$work/full/cgroup" 950000 1000000
# A directory where memory.reclaim should be, so writing it fails the way a kernel refusal does.
mkdir "$work/full/cgroup/memory.reclaim"
start_guard full "[f]akeserver-$$" "$work/full/cache"
wait_for gone "$guard_pid" || fail "full: the guard did not fire and exit"
code=0
wait "$guard_pid" || code=$?
(( code == 0 )) || fail "full: the guard exited $code"
wait_for gone "$server" || fail "full: the server outlived the guard"
rows full "len(rows) == 3 and rows[1]['trim'] == {'fadvise': $fadvise, 'reclaim': 'EISDIR'} and 'fired' not in rows[1]
           and rows[2]['fired'] is True and 'trim' not in rows[2]"
report=$work/full/run/memory_guard.txt
reading='^fired [0-9T:-]+Z: memory.current 0.0 GiB is 95.0% of memory.max 0.0 GiB, at or over the kill line of 93%'
head -n 1 "$report" | grep -qE "$reading after a page cache trim\$" \
    || fail "full: the report's first line is not the reading: $(head -n 1 "$report")"
grep -qx 'kernel 99' "$report" || fail "full: memory.stat is not in the report whole"
grep -q '^\$ ps -eo pid,ppid,stat,rss,pcpu,etime,args --sort=-rss' "$report" || fail "full: no process table"
grep -qx "$server" "$report" || fail "full: pgrep did not list the server"
grep -qx "py-spy dump --nonblocking --pid $server" "$report" || fail "full: no stack of the server"
grep -qF "\$ pkill -9 -f '[f]akeserver-$$' (exit 0)" "$report" || fail "full: the kill is not recorded"
python3 -I - "$work/full/run/progress.jsonl" <<'EOF' || fail "full: no mem_guard phase in progress.jsonl"
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
assert [row["phase"] for row in rows] == ["mem_guard"], rows
assert rows[0]["detail"].startswith("fired ") and type(rows[0]["t"]) is int, rows
EOF
echo "ok: over the kill line after a trim the guard reports, kills the server's processes, and exits"

cgroup_v2 "$work/max/cgroup" 990000 max
start_guard max '[f]akeserver-none' "$work/max/cache"
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
start_guard v1 '[f]akeserver-none'
wait_for has_lines "$work/v1/run/memory.jsonl" 3 || fail "v1: fewer than 3 lines in 10 s"
stop v1
rows v1 "rows[0] == $missing_controls and 'warn' in rows[1] and rows[2] == {'t': rows[2]['t'], 'current': 500,
         'max': None, 'anon': 20, 'file': 10, 'active_file': 60, 'inactive_file': 70, 'shmem': 50, 'file_mapped': 30,
         'file_dirty': 40, 'unevictable': 80}"
echo "ok: cgroup v1 counters map to the v2 names, and its no-limit value means no limit"

start_guard missing '[f]akeserver-none'
wait_for grep -q 'poll failed: FileNotFoundError' "$work/missing/mem_guard.log" || fail "missing: no error logged"
sleep 0.3
stop missing
echo "ok: a cgroup it cannot read is logged, and the guard keeps polling"

mkdir -p "$work/orphan/run"
cgroup_v2 "$work/orphan/cgroup" 100000 1000000
MEM_GUARD_CGROUP=$work/orphan/cgroup MEM_GUARD_POLL_S=0.1 bash -c 'python3 -I "$0" "$1" "$2" >"$3" 2>&1 &
echo $! >"$4"
sleep 2' "$guard" "$work/orphan/run" '[f]akeserver-none' "$work/orphan/mem_guard.log" "$work/orphan/pid"
orphan=$(<"$work/orphan/pid")
pids+=("$orphan")
wait_for gone "$orphan" || fail "orphan: the guard outlived its parent"
has_lines "$work/orphan/run/memory.jsonl" 3 || fail "orphan: the guard did not run while its parent lived"
echo "ok: a guard whose parent dies without stopping it exits"

hub=$work/hf/hub/models--org--model
mkdir -p "$hub/blobs" "$hub/snapshots/abc"
printf '%01000d' 0 >"$hub/blobs/aaa"
printf '%0500d' 0 >"$hub/blobs/bbb"
echo outside >"$work/outside.txt"
ln -s ../../blobs/aaa "$hub/snapshots/abc/model.safetensors"
ln -s ../../blobs/bbb "$hub/snapshots/abc/config.json"
ln -s "$work/outside.txt" "$hub/snapshots/abc/outside.txt"
ln -s ../../blobs/missing "$hub/snapshots/abc/dangling"
MEM_GUARD_CGROUP=$work/low/cgroup python3 -I "$guard" --drop-cache "$work/hf" "$work/nonexistent" >"$work/drop.out" \
    || fail "drop-cache exited non-zero"
done_re='(dropped|found \(no posix_fadvise here, so none dropped\))'
grep -qE "^drop-cache: $done_re 3 files, 0.0 GB, 1 failed; memory.current 0.0 GiB -> 0.0 GiB\$" \
    "$work/drop.out" || fail "drop-cache summary: $(cat "$work/drop.out")"
MEM_GUARD_CGROUP=$work/missing python3 -I "$guard" --drop-cache "$work/hf" >"$work/drop.out" \
    || fail "drop-cache exited non-zero without a cgroup"
grep -q 'memory.current unreadable -> unreadable$' "$work/drop.out" || fail "drop-cache: $(cat "$work/drop.out")"
echo "ok: --drop-cache counts each file once through the snapshot links, a dangling link as failed, and exits 0"
