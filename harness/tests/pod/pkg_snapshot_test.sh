#!/usr/bin/env bash
# Saves and restores the fake sglang package with pkg_snapshot.sh.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
snapshot="$here/../../pod/pkg_snapshot.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -R "$here/fake_site" "$work/site"
export PYTHONPATH="$work/site" PKG_SNAPSHOT_DIR="$work/snap"
mkdir -p "$work/snap"
core="$work/site/sglang/core.py"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

if bash "$snapshot" restore sglang 2>/dev/null; then
    fail "restore without a snapshot must fail"
fi
echo "ok: restore needs a snapshot"

bash "$snapshot" save sglang
echo 'BEHAVIOR = "patched"' >"$core"
echo 'stray = 1' >"$work/site/sglang/added.py"
bash "$snapshot" restore sglang
grep -q upstream "$core" || fail "restore did not bring the file back"
[[ ! -e $work/site/sglang/added.py ]] || fail "restore left a file the snapshot does not have"
echo "ok: restore puts back the saved tree exactly"

echo 'BEHAVIOR = "patched"' >"$core"
bash "$snapshot" save sglang
bash "$snapshot" restore sglang
grep -q upstream "$core" || fail "a second save replaced the first copy"
echo "ok: the first save is the one kept"

if bash "$snapshot" save no_such_package 2>/dev/null; then
    fail "an unknown package must fail"
fi
echo "ok: an unknown package fails"
