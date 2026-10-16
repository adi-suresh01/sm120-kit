#!/usr/bin/env bash
# Runs serve_variant.sh on this machine against fake_server.py and the fake sglang and fakeinfer packages: a
# healthy boot with the quality protocol and a patch to each package, a fatal log line, a hung boot, a failed smoke
# test, and a server that dies mid-evaluation. Also checks restores between runs, the hooks, and progress.jsonl.
# Then vllm variants with quality, through gsm8k_chat.py on a fixture and fake_vllm.py: right answers, garbage
# answers, HTTP 500s, a retried failure, a max_model_len too short to answer in, a fixture hash mismatch, and a bench
# run that drops a request. Last, an issue whose hooks.sh defines post_smoke: alone, failing, after quality, under
# a server that dies, and leaving a process running behind it.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pod="$here/../../pod"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cp -R "$here/fake_site" "$work/site"
mkdir -p "$work/bin" "$work/issue/patches" "$work/snap"
cp "$here/demo.diff" "$work/issue/patches/demo.diff"
# A -p2 patch to the second package, so each patch's own package and strip are exercised.
cat >"$work/issue/patches/infer.diff" <<'EOF'
--- a/pkgs/fakeinfer/core.py
+++ b/pkgs/fakeinfer/core.py
@@ -1 +1 @@
-KERNEL = "upstream"
+KERNEL = "patched"
EOF
printf '#!/usr/bin/env bash\necho 0\n' >"$work/bin/nvidia-smi"
# macOS has no timeout(1); the outer bound is not what these cases test.
if ! command -v timeout >/dev/null; then
    cat >"$work/bin/timeout" <<'EOF'
#!/usr/bin/env bash
while [[ $1 == -* ]]; do
    [[ $1 == -k || $1 == -s ]] && shift
    shift
done
shift
exec "$@"
EOF
fi
# The cleanup's `pkill -9 -f` must never run on this machine, where the pattern could match unrelated processes.
cat >"$work/bin/pkill" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$WORK/pkill.log"
EOF
# macOS has no setsid(1). The stand-in has to start a new session too, since serve_and_smoke.sh ends POST_SMOKE by
# its process group.
if ! command -v setsid >/dev/null; then
    cat >"$work/bin/setsid" <<'EOF'
#!/usr/bin/env bash
exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@"
EOF
fi
cat >"$work/bin/vllm" <<EOF
#!/usr/bin/env bash
exec python3 -I '$here/fake_vllm.py' "\$@"
EOF
chmod +x "$work/bin/"*
export PATH="$work/bin:$PATH" PYTHONPATH="$work/site" PKG_SNAPSHOT_DIR="$work/snap" WORK="$work"
export BOOT_POLL_S=0.2 POST_POLL_S=0.2 STOP_POLL_S=0.1
cat >"$work/issue/hooks.sh" <<'EOF'
pre_launch() { printf '%s|%s|%s\n' "$1" "${FAKE_ENV:-}" "$(cat "$WORK/site/fakeinfer/core.py")" >>"$WORK/pre.txt"; }
post_stop() { printf '%s\n' "$1" >>"$WORK/post.txt"; }
EOF

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# A server flag with quotes, a backslash, and a newline, which the launch phase's detail must carry intact.
note=$'--note=say "hi"\nback\\slash'

# Writes a variant file shaped like `harness exec` renders one, serving $package and restoring $snapshot_pkgs.
# Args: name server gsm8k env [patch:package:strip...]; server is the fake server's mode, then any flags for it.
package=sglang snapshot_pkgs="'sglang' 'fakeinfer'"
write_variant() {
    local name=$1 server=$2 gsm8k=$3 env=$4 files="" pkgs="" strips="" spec file pkg strip
    shift 4
    for spec in "$@"; do
        IFS=: read -r file pkg strip <<<"$spec"
        files+=" '$file'" pkgs+=" '$pkg'" strips+=" '$strip'"
    done
    cat >"$work/$name.sh" <<EOF
VARIANT='$name'
PACKAGE='$package'
MODEL='fake/model'
PORT='$(free_port)'
FAIL_PATTERN='illegal memory access'
BOOT_TIMEOUT='1'
CONCURRENCY='2'
GSM8K='$gsm8k'
SERVER_CMD=('python3' '$here/fake_server.py' $server '$note')
VARIANT_ENV=(${env:+"'$env'"})
BENCH=('1:8' '16:64')
SNAPSHOT_PKGS=($snapshot_pkgs)
PATCH_FILES=($files)
PATCH_PKGS=($pkgs)
PATCH_STRIPS=($strips)
EOF
}

issue="$work/issue"
serve() {
    bash "$pod/serve_variant.sh" "$issue" "$work/$1.sh" "$work/results" >"$work/$1.out" 2>&1 \
        || fail "serve_variant.sh $1 exited non-zero"
}

fail() {
    echo "FAIL: $*" >&2
    cat "$work"/*.out >&2 2>/dev/null || true
    exit 1
}

# Checks one verdict.json field against a Python literal.
field() {
    python3 -I - "$work/results/$1/verdict.json" "$2" "$3" <<'EOF' || fail "$1: $2"
import ast
import json
import sys

path, key, want = sys.argv[1:]
got = json.load(open(path))[key]
if got != ast.literal_eval(want):
    sys.exit(f"{key}: got {got!r}, want {want}")
EOF
}

# Checks a run's progress.jsonl: every line is JSON, the phases come in the given order, the launch detail is the
# exact server command, and the last line is verdict.json itself. Args: variant "phase phase ..."
phases() {
    local launch
    # shellcheck source=/dev/null
    launch=$(source "$work/$1.sh" && printf '%s' "${SERVER_CMD[*]} --host 127.0.0.1 --port $PORT")
    python3 -I - "$work/results/$1" "$2" "$launch" <<'EOF' || fail "$1: progress.jsonl"
import json
import sys

run_dir, want, launch = sys.argv[1:]
lines = open(f"{run_dir}/progress.jsonl", encoding="utf-8").read().splitlines()
rows = [json.loads(line) for line in lines]
got = [row.get("phase") for row in rows[:-1]]
if got != want.split():
    sys.exit(f"phases: got {got}, want {want.split()}")
if not all(type(row["t"]) is int for row in rows[:-1]):
    sys.exit("every phase needs an integer t")
if lines[-1] != open(f"{run_dir}/verdict.json", encoding="utf-8").read().rstrip("\n"):
    sys.exit("the last line is not verdict.json")
details = [row.get("detail") for row in rows if row.get("phase") == "launch"]
if details != [launch]:
    sys.exit(f"launch detail: got {details!r}, want {launch!r}")
EOF
}

write_variant healthy ok 50 'FAKE_ENV=from the variant' demo:sglang:1 infer:fakeinfer:2
serve healthy
field healthy verdict "'OK'"
field healthy fault None
field healthy gsm8k 0.97
field healthy accept 2.9374
field healthy tok_s "{'1': 100.0, '16': 1600.0}"
field healthy verdict_error None
grep -q 'patched' "$work/site/sglang/core.py" || fail "sglang patch not applied"
grep -q 'patched' "$work/site/fakeinfer/core.py" || fail "fakeinfer patch not applied at -p2"
grep -q "^$work/results/healthy|from the variant|" "$work/pre.txt" || fail "pre_launch did not see the run dir and env"
[[ $(wc -l <"$work/post.txt") -eq 1 ]] || fail "post_stop should run once"
phases healthy "restore patch patch idle_wait launch boot_ok smoke_ok quality_start quality_done stop"
echo "ok: healthy boot with quality, a patch to each of two packages, hooks, and progress"

write_variant faulty fatal "" ""
serve faulty
field faulty verdict "'BOOT_FAIL'"
field faulty fault "'CUDA error: an illegal memory access was encountered'"
field faulty where "'driver.py\", line 331, in __call__'"
grep -q 'upstream' "$work/site/sglang/core.py" || fail "the next run did not start from the pristine sglang"
[[ $(tail -n 1 "$work/pre.txt") == "$work/results/faulty||KERNEL = \"upstream\"" ]] \
    || fail "fakeinfer was still patched when the next variant launched"
phases faulty "restore idle_wait launch boot_fail stop"
echo "ok: a fatal log line ends the boot as BOOT_FAIL, and both packages were restored before it"

write_variant stuck hang "" ""
serve stuck
field stuck verdict "'TIMEOUT'"
field stuck boot_s 1
phases stuck "restore idle_wait launch boot_fail stop"
echo "ok: a hung boot ends as TIMEOUT"

write_variant smokefail smoke_fail 50 ""
serve smokefail
field smokefail verdict "'SMOKE_FAIL'"
[[ ! -e $work/results/smokefail/server.post.log ]] || fail "quality ran after a failed smoke test"
phases smokefail "restore idle_wait launch boot_ok smoke_fail stop"
echo "ok: a failed chat request ends as SMOKE_FAIL without running quality"

write_variant postfail post_die 50 'FAKE_EVAL=hang'
serve postfail
field postfail verdict "'POST_FAIL'"
grep -q 'post_exit=server_exit' "$work/results/postfail/verdict.json" || fail "postfail: not stopped for server_exit"
phases postfail "restore idle_wait launch boot_ok smoke_ok quality_start quality_done stop"
grep -q '"detail": "post_exit=server_exit"' "$work/results/postfail/progress.jsonl" || fail "quality_done detail"
# The eval client is a grandchild of serve_and_smoke; it must not keep retrying the dead server.
# shellcheck source=/dev/null
post_port=$(source "$work/postfail.sh" && echo "$PORT")
if pgrep -f "sglang.test.run_eval .*--port $post_port" >/dev/null; then
    fail "the eval client outlived its server"
fi
echo "ok: a server that dies mid-evaluation ends as POST_FAIL and takes the eval client down"

# The vllm variants ask for 3 of the fixture's 4 test questions; its train split has 6 rows, of which 5 are shots.
package=vllm snapshot_pkgs="'vllm'"
sha256() {
    python3 -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
}
# A copy, so a client that wrote its cache would not write into the repo.
cp -R "$here/gsm8k" "$work/gsm8k"
GSM8K_TRAIN_SHA256=$(sha256 "$work/gsm8k/train.jsonl")
GSM8K_TEST_SHA256=$(sha256 "$work/gsm8k/test.jsonl")
export GSM8K_DIR="$work/gsm8k" GSM8K_TRAIN_SHA256 GSM8K_TEST_SHA256

# Checks a run's gsm8k_responses.jsonl: one row for each of the given fixture questions (default the 3 asked), in
# any order, since rows land as replies do, and in every row the fields of a Python dict literal.
# Args: variant fields [indices]
responses() {
    python3 -I - "$work/results/$1/gsm8k_responses.jsonl" "$GSM8K_DIR/test.jsonl" "$2" "${3:-0 1 2}" <<'EOF' \
        || fail "$1: responses"
import ast
import json
import sys

path, fixture, want, indices = sys.argv[1:]
rows = sorted((json.loads(line) for line in open(path, encoding="utf-8")), key=lambda row: row["index"])
questions = [json.loads(line)["question"] for line in open(fixture, encoding="utf-8")]
expected = [(int(index), questions[int(index)]) for index in indices.split()]
if [(row["index"], row["question"]) for row in rows] != expected:
    sys.exit(f"want rows for fixture questions {indices}, got {rows}")
for row in rows:
    bad = {key: row[key] for key, value in ast.literal_eval(want).items() if row[key] != value}
    if bad:
        sys.exit(f"row {row['index']}: {bad}")
EOF
}

# Checks the fields of one gsm8k_responses.jsonl row against a Python dict literal. Args: variant index fields
row() {
    python3 -I - "$work/results/$1/gsm8k_responses.jsonl" "$2" "$3" <<'EOF' || fail "$1: row $2"
import ast
import json
import sys

path, index, want = sys.argv[1:]
row = next(row for row in map(json.loads, open(path, encoding="utf-8")) if row["index"] == int(index))
bad = {key: row[key] for key, value in ast.literal_eval(want).items() if row[key] != value}
if bad:
    sys.exit(f"row {index}: {bad}")
EOF
}

write_variant vllm-correct "ok --fake-gsm8k correct" 3 ""
serve vllm-correct
field vllm-correct verdict "'OK'"
field vllm-correct gsm8k 1.0
field vllm-correct accept None
field vllm-correct tok_s "{'1': 100.0, '16': 1600.0}"
field vllm-correct verdict_error None
# The fixture's references include a thousands comma and a negative number.
responses vllm-correct "{'correct': True, 'finish_reason': 'stop', 'error': None}"
grep -q 'max_tokens 4096 (max_model_len 8192)' "$work/results/vllm-correct/gsm8k.txt" || fail "default max_tokens"
for c in 1 16; do
    [[ -s $work/results/vllm-correct/bench_c$c.json ]] || fail "no bench_c$c.json"
done
phases vllm-correct "restore idle_wait launch boot_ok smoke_ok quality_start quality_done stop"
echo "ok: vllm quality scores a server with the right answers 1.0, and both bench runs report tok/s"

write_variant vllm-garbage "ok --fake-gsm8k garbage --max-model-len 3000" 3 ""
serve vllm-garbage
field vllm-garbage verdict "'OK'"
field vllm-garbage gsm8k 0.0
responses vllm-garbage "{'correct': False, 'parsed': None, 'finish_reason': 'length', 'completion_tokens': 952}"
echo "ok: a server answering in commas completes with 0.0, each reply capped at max_model_len 3000 - 2048 tokens"

write_variant vllm-error "ok --fake-gsm8k error" 3 ""
serve vllm-error
field vllm-error verdict "'POST_FAIL'"
field vllm-error gsm8k None
grep -q '3/3 requests failed against the server; first: HTTP 500' "$work/results/vllm-error/gsm8k.txt" \
    || fail "vllm-error: no failure line"
responses vllm-error "{'outcome': 'failed', 'attempts': 3, 'response': None}"
[[ ! -e $work/results/vllm-error/bench.jsonl ]] || fail "vllm-error: bench ran after GSM8K failed"
echo "ok: GSM8K requests that keep failing with 500 fail the quality run after 3 tries each, with no score"

write_variant vllm-flaky "ok --fake-gsm8k flaky" 3 ""
serve vllm-flaky
field vllm-flaky verdict "'OK'"
field vllm-flaky gsm8k 1.0
responses vllm-flaky "{'outcome': 'answered', 'attempts': 2, 'correct': True}"
grep -q '^gsm8k: answered=3 .* retried=3$' "$work/results/vllm-flaky/gsm8k.txt" || fail "vllm-flaky: retries not shown"
echo "ok: a request that fails once is retried, and the retries show"

write_variant vllm-slow "ok --fake-gsm8k slow" 3 "GSM8K_TIMEOUT_S=1"
serve vllm-slow
field vllm-slow verdict "'OK'"
field vllm-slow gsm8k 0.667
grep -q '^gsm8k: answered=2 correct=2 .* timed_out=1 failed=0 retried=0$' "$work/results/vllm-slow/gsm8k.txt" \
    || fail "vllm-slow: summary"
row vllm-slow 1 "{'outcome': 'timeout', 'attempts': 1, 'correct': False}"
row vllm-slow 2 "{'outcome': 'answered', 'correct': True}"
echo "ok: a request that times out is scored wrong once, not retried, and the run still scores"

write_variant vllm-fault "ok --fake-gsm8k fault" 3 ""
serve vllm-fault
field vllm-fault verdict "'POST_FAIL'"
field vllm-fault gsm8k None
grep -q 'post_exit=pattern' "$work/results/vllm-fault/verdict.json" || fail "vllm-fault: not stopped for the pattern"
responses vllm-fault "{'outcome': 'answered', 'correct': True}" "0 2"
echo "ok: a server that faults mid-GSM8K ends as POST_FAIL with every reply it gave already on disk"

write_variant vllm-short "ok --fake-gsm8k correct --max-model-len 2200" 3 ""
serve vllm-short
field vllm-short verdict "'POST_FAIL'"
grep -q 'max_tokens would be 152' "$work/results/vllm-short/gsm8k.txt" || fail "vllm-short: no max_tokens error"
[[ ! -e $work/results/vllm-short/gsm8k_responses.jsonl ]] || fail "vllm-short: GSM8K requests were sent"
echo "ok: a max_model_len that leaves under 256 tokens to answer in fails before any GSM8K request"

write_variant vllm-badsha "ok --fake-gsm8k correct" 3 "GSM8K_TEST_SHA256=0000"
serve vllm-badsha
field vllm-badsha verdict "'POST_FAIL'"
grep -q "test.jsonl has sha256 $GSM8K_TEST_SHA256, want 0000" "$work/results/vllm-badsha/gsm8k.txt" \
    || fail "vllm-badsha: no hash error"
echo "ok: a GSM8K file whose hash is not the pinned one fails the quality run"

write_variant vllm-drop "ok --fake-gsm8k correct" 3 "FAKE_BENCH=drop"
serve vllm-drop
field vllm-drop verdict "'POST_FAIL'"
field vllm-drop gsm8k 1.0
field vllm-drop tok_s "{'1': 100.0}"
grep -q 'bench c=1: 7/8 requests completed' "$work/results/vllm-drop/server.post.log" \
    || fail "vllm-drop: no bench error"
echo "ok: a bench run that completes fewer requests than it sent fails the quality run and keeps its speed"

# The hook writes what it was called with and saw, asks the live server, adds a phase, and appends "finished" last.
# The variant env picks its mode: HOOK=fail fails a command in its middle, HOOK=hang waits like a client that keeps
# retrying, and HOOK=background leaves a process running behind it.
issue="$work/hooked"
mkdir -p "$issue"
cat >"$issue/hooks.sh" <<'EOF'
post_smoke() {
    if [[ ${HOOK:-} == hang ]]; then
        sleep 29.123
    fi
    if [[ ${HOOK:-} == background ]]; then
        sleep 28.456 &
    fi
    printf '%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$VARIANT" "${HOOK:-}" >"$1/hook.txt"
    curl -sf -w '\n' "http://127.0.0.1:$2/health" >>"$1/hook.txt"
    if [[ -f $1/gsm8k.txt ]]; then
        grep '^Score:' "$1/gsm8k.txt" >>"$1/hook.txt"
    fi
    progress hook_step "from the hook"
    [[ ${HOOK:-} != fail ]]
    echo finished >>"$1/hook.txt"
}
EOF

write_variant hook-ok ok "" "HOOK=ok"
serve hook-ok
field hook-ok verdict "'OK'"
field hook-ok gsm8k None
# shellcheck source=/dev/null
hook_port=$(source "$work/hook-ok.sh" && echo "$PORT")
[[ $(head -n 1 "$work/results/hook-ok/hook.txt") == "$work/results/hook-ok|$hook_port|fake/model|hook-ok|ok" ]] \
    || fail "hook-ok: post_smoke's arguments, the variant's variables, or its env"
grep -qxF '{"ok": true}' "$work/results/hook-ok/hook.txt" || fail "hook-ok: the server did not answer the hook"
grep -qx finished "$work/results/hook-ok/hook.txt" || fail "hook-ok: the hook did not finish"
grep -qF '"detail": "from the hook"' "$work/results/hook-ok/progress.jsonl" || fail "hook-ok: progress detail"
phases hook-ok "restore idle_wait launch boot_ok smoke_ok quality_start hook_start hook_step quality_done stop"
echo "ok: post_smoke runs on the live server with its arguments, the variant's variables and env, and progress"

write_variant hook-fail ok "" "HOOK=fail"
serve hook-fail
field hook-fail verdict "'POST_FAIL'"
grep -q 'post_exit=1' "$work/results/hook-fail/verdict.json" || fail "hook-fail: not stopped for the hook's exit"
if grep -qx finished "$work/results/hook-fail/hook.txt"; then
    fail "hook-fail: the hook ran on past its failing command"
fi
echo "ok: post_smoke stops at its first failing command, and the run is POST_FAIL"

write_variant hook-quality "ok --fake-gsm8k correct" 3 "HOOK=ok"
serve hook-quality
field hook-quality verdict "'OK'"
field hook-quality gsm8k 1.0
grep -qx 'Score: 1.000' "$work/results/hook-quality/hook.txt" || fail "hook-quality: post_smoke ran before quality"
phases hook-quality "restore idle_wait launch boot_ok smoke_ok quality_start hook_start hook_step quality_done stop"
echo "ok: with quality, post_smoke runs after it"

write_variant hook-qfail "ok --fake-gsm8k correct" 3 "FAKE_BENCH=drop"
serve hook-qfail
field hook-qfail verdict "'POST_FAIL'"
grep -qx finished "$work/results/hook-qfail/hook.txt" || fail "hook-qfail: post_smoke did not run after quality failed"
echo "ok: post_smoke still runs after quality fails, and the run is POST_FAIL"

write_variant hook-die post_die "" "HOOK=hang"
serve hook-die
field hook-die verdict "'POST_FAIL'"
grep -q 'post_exit=server_exit' "$work/results/hook-die/verdict.json" || fail "hook-die: not stopped for server_exit"
if pgrep -fx 'sleep 29.123' >/dev/null; then
    fail "hook-die: what the hook started outlived its server"
fi
echo "ok: a server that dies under post_smoke ends as POST_FAIL and takes the hook's processes down"

write_variant hook-bg ok "" "HOOK=background"
serve hook-bg
field hook-bg verdict "'OK'"
grep -qx finished "$work/results/hook-bg/hook.txt" || fail "hook-bg: the hook did not finish"
if pgrep -fx 'sleep 28.456' >/dev/null; then
    fail "hook-bg: a process the hook backgrounded outlived the run"
fi
echo "ok: a process post_smoke leaves running behind it is ended with the run"

[[ $(wc -l <"$work/results/verdicts.jsonl") -eq 20 ]] || fail "verdicts.jsonl should hold one line per run"
cmp -s <(tail -n 1 "$work/results/verdicts.jsonl") "$work/results/hook-bg/verdict.json" || fail "verdict.json differs"
if pgrep -f "$here/fake_server.py" >/dev/null; then
    fail "a fake server outlived its run"
fi
[[ $(grep -cxF -- '-9 -f [s]glang[.]launch_server|[s]glang::' "$work/pkill.log") -eq 5 ]] \
    || fail "each sglang run should clean up the server pattern once"
[[ $(grep -cxF -- '-9 -f [v]llm serve|[V]LLM::' "$work/pkill.log") -eq 15 ]] \
    || fail "each vllm run should clean up the server pattern once"
echo "ok: one verdict line per run, one cleanup per run, and no server left behind"
