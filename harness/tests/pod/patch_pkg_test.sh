#!/usr/bin/env bash
# Applies demo.diff to the fake sglang package with patch_pkg.sh, and checks that each way it can fail says why.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
patch_pkg="$here/../../pod/patch_pkg.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -R "$here/fake_site" "$work/site"
export PYTHONPATH="$work/site"
diff="$here/demo.diff"

fail() {
    echo "FAIL: $*" >&2
    cat "$work/err" >&2 2>/dev/null || true
    exit 1
}

bash "$patch_pkg" apply "$diff" sglang >/dev/null
grep -q patched "$work/site/sglang/core.py" || fail "apply"
[[ $(bash "$patch_pkg" apply "$diff" sglang) == "already applied" ]] || fail "a second apply"
bash "$patch_pkg" revert "$diff" sglang >/dev/null
grep -q upstream "$work/site/sglang/core.py" || fail "revert"
echo "ok: applies, notices it is applied, and reverts"

# Only python3 on PATH, so patch(1) is missing.
mkdir -p "$work/nopatch"
ln -s "$(command -v python3)" "$work/nopatch/python3"
if PATH="$work/nopatch" "$BASH" "$patch_pkg" apply "$diff" sglang 2>"$work/err"; then
    fail "apply without patch(1) must fail"
fi
grep -q 'patch(1) is not installed' "$work/err" || fail "a missing patch(1) is not named"
echo "ok: a missing patch(1) is named"

if bash "$patch_pkg" apply "$work/no-such.diff" sglang 2>"$work/err"; then
    fail "apply of a missing file must fail"
fi
grep -q "cannot read patch file: $work/no-such.diff" "$work/err" || fail "a missing patch file is not named"
echo "ok: an unreadable patch file is named"

echo 'BEHAVIOR = "drifted"' >"$work/site/sglang/core.py"
if bash "$patch_pkg" apply "$diff" sglang 2>"$work/err"; then
    fail "a drifted package must fail"
fi
grep -qiE '1 out of 1 hunks? failed' "$work/err" || fail "the dry run's reason is hidden"
grep -q 'does not apply cleanly' "$work/err" || fail "no summary"
grep -q drifted "$work/site/sglang/core.py" || fail "a failed apply touched the package"
echo "ok: a patch that does not apply shows the dry run's reason and changes nothing"
