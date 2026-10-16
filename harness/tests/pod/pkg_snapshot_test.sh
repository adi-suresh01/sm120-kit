#!/usr/bin/env bash
# Saves and restores the fake sglang and fakeinfer packages with pkg_snapshot.sh, including restores that fail or
# are cut off halfway.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
snapshot="$here/../../pod/pkg_snapshot.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -R "$here/fake_site" "$work/site"
export PYTHONPATH="$work/site" PKG_SNAPSHOT_DIR="$work/snap"
core="$work/site/sglang/core.py"
infer="$work/site/fakeinfer/core.py"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

importable() {
    python3 -c "import $1.core" 2>/dev/null
}

if bash "$snapshot" restore sglang 2>/dev/null; then
    fail "restore without a snapshot must fail"
fi
echo "ok: restore needs a snapshot"

bash "$snapshot" save sglang fakeinfer
echo 'BEHAVIOR = "patched"' >"$core"
echo 'KERNEL = "patched"' >"$infer"
echo 'stray = 1' >"$work/site/sglang/added.py"
bash "$snapshot" restore sglang fakeinfer
grep -q upstream "$core" || fail "restore did not bring the file back"
grep -q upstream "$infer" || fail "restore did not bring the second package back"
[[ ! -e $work/site/sglang/added.py ]] || fail "restore left a file the snapshot does not have"
[[ -z $(find "$work/site" -name '*.new' -o -name '*.old') ]] || fail "restore left its scratch copies"
echo "ok: restore puts back every saved tree exactly"

echo 'BEHAVIOR = "patched"' >"$core"
bash "$snapshot" save sglang
bash "$snapshot" restore sglang
grep -q upstream "$core" || fail "a second save replaced the first copy"
echo "ok: the first save is the one kept"

# A copy that fails partway (full disk, I/O error) must leave the installed package as it was.
mkdir -p "$work/failbin"
printf '#!/usr/bin/env bash\necho "cp: simulated failure" >&2\nexit 1\n' >"$work/failbin/cp"
chmod +x "$work/failbin/cp"
echo 'BEHAVIOR = "patched"' >"$core"
if PATH="$work/failbin:$PATH" bash "$snapshot" restore sglang 2>/dev/null; then
    fail "a failed copy must fail the restore"
fi
importable sglang || fail "a failed restore left sglang unimportable"
grep -q patched "$core" || fail "a failed restore changed the installed package"
echo "ok: a failed restore leaves the installed package in place"

# Killed between its two renames: the package is gone from its path and only the scratch copies remain.
mv "$work/site/sglang" "$work/site/sglang.old"
mkdir -p "$work/site/sglang.new"
echo 'half = 1' >"$work/site/sglang.new/core.py"
! importable sglang || fail "the cut-off restore should have left sglang missing"
bash "$snapshot" restore sglang
importable sglang || fail "restore did not repair a cut-off restore"
grep -q upstream "$core" || fail "the repaired package is not the snapshot"
[[ -z $(find "$work/site" -name '*.new' -o -name '*.old') ]] || fail "repair left scratch copies"
echo "ok: the next restore repairs one cut off between its renames"

if bash "$snapshot" save no_such_package 2>/dev/null; then
    fail "an unknown package must fail"
fi
echo "ok: an unknown package fails"

echo 'x = 1' >"$work/site/lonely.py"
if bash "$snapshot" save lonely 2>/dev/null; then
    fail "a single-file module must be refused"
fi
[[ ! -e $PKG_SNAPSHOT_DIR/lonely.pristine ]] || fail "a single-file module was snapshotted"
echo "ok: a single-file module is refused instead of copying site-packages"
