"""Downloads the published block-FP8 [32, 32] tables the bench compares against, from pinned commits, into
<out>/tables/<set>/ under their upstream names: sglang#39872's RTX PRO 6000 Server Edition tables (head 7025bba, not
merged) and sglang#45263's RTX PRO 5000 tables (merged 11-10; main at c2578f2)."""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SETS = {
    "pr39872": (
        "7025bbae609dfaeff93f424ff1822fe22b71277f",
        "python/sglang/kernels/ops/quantization/configs",
        "NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition",
        [(1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 288), (25600, 6144)],
    ),
    "pro5000": (
        "c2578f28cfaece96167765ed93cab9a484f6fe95",
        "python/sglang/kernels/ops/gemm/configs",
        "NVIDIA_RTX_PRO_5000_72GB_Blackwell",
        [(1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 288), (25600, 6144)]
        + [(32768, 1280), (5120, 8192)],
    ),
}


def main():
    out = Path(sys.argv[1]) / "tables"
    missing = 0
    for name, (sha, directory, device, shapes) in SETS.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        for n, k in shapes:
            file = f"N={n},K={k},device_name={device},dtype=fp8_w8a8,block_shape=[32, 32].json"
            # A PR's head commit is served from the base repository too, so a deleted fork does not lose it.
            path = urllib.parse.quote(f"{directory}/{file}")
            url = f"https://raw.githubusercontent.com/sgl-project/sglang/{sha}/{path}"
            try:
                table = json.load(urllib.request.urlopen(url, timeout=30))
            except Exception as e:  # the arm then reports "no table" for this shape; the other sets still land
                print(f"{name} {n}x{k}: {e}", file=sys.stderr)
                missing += 1
                continue
            (out / name / file).write_text(json.dumps(table, indent=4) + "\n")
            print(f"{name} {n}x{k}: M {sorted(int(m) for m in table)}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
