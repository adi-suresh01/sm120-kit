"""Builds 12K, 35K, and 120K-token prompts from real prose with the model's own tokenizer.

The reporter saw the fault depend on content, not only length, so random token ids would not be a faithful repro.
Usage: build_prompts.py <hf_model> <revision> <out_dir>
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# War and Peace: about 750K tokens, so one book covers every length without stitching texts together.
SOURCE = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
LENGTHS = {"12k": 12_000, "35k": 35_000, "120k": 120_000}
INSTRUCTION = "Below is a long excerpt from a novel. Summarize what happens in it in three sentences.\n\n"


def main() -> None:
    model, revision, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE, timeout=120) as resp:
        raw = resp.read()
    text = raw.decode("utf-8-sig")
    start = text.index("*** START OF")
    body = text[text.index("\n", start) + 1 : text.index("*** END OF")]
    # Skip the table of contents so every prompt is prose.
    body = body[body.index("\u201cWell, Prince, so Genoa") :]
    tok = Tokenizer.from_file(hf_hub_download(model, "tokenizer.json", revision=revision))
    ids = tok.encode(body, add_special_tokens=False).ids
    manifest = {"source": SOURCE, "source_sha256": hashlib.sha256(raw).hexdigest(), "body_tokens": len(ids)}
    for name, n in LENGTHS.items():
        if len(ids) < n:
            sys.exit(f"source has {len(ids)} tokens, fewer than {n}")
        prompt = INSTRUCTION + tok.decode(ids[:n])
        (out / f"{name}.txt").write_text(prompt)
        manifest[name] = len(tok.encode(prompt, add_special_tokens=False).ids)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
