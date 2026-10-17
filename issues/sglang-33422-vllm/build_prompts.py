"""Builds the warmup, W1, and W2 prompts from real prose, sized by the fully templated token count.

Each prompt is counted the way both servers count it: the checkpoint's own chat encoding
(encoding/encoding_dsv4.py, chat mode, one user message) through the PreTrainedTokenizerFast both servers load,
with a nonce of the client's shape in front. The client still computes tok/s from the server's usage.prompt_tokens.
Usage: build_prompts.py <hf_model> <revision> <out_dir>
"""

import hashlib
import importlib.util
import json
import sys
import urllib.request
import uuid
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast

# War and Peace: 775K tokens with this tokenizer, enough for the 238K prompt without stitching texts together. The
# hash keeps the prompts identical on every pod, so the two engines read the same text.
SOURCE = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
SOURCE_SHA256 = "2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b"
# Every W1 prompt shares one text, so only the nonce keeps the prefix caches from serving one request from another.
NONCE_TEMPLATE = "Request {}.\n\n"
INSTRUCTION = "Below is a long excerpt from a novel. Summarize what happens in it in three sentences.\n\n"
# W2 at 238K is the prompt behind hermia's ~12.5K tok/s, and stays under vLLM's --max-model-len 262144 with its
# 40 output tokens.
TARGETS = {"warmup": 8_000, "w1": 90_000, "w2": 238_000}


def main() -> None:
    model, revision, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE, timeout=120) as resp:
        raw = resp.read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        sys.exit(f"{SOURCE} changed; its sha256 is no longer {SOURCE_SHA256}")
    # The book has CRLF line ends, which vllm-56771's builder counted and the client's read_text() turned into LF
    # before sending: that, not the template, is why its "12k" reached the server as 10,740 tokens.
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    start = text.index("*** START OF")
    body = text[text.index("\n", start) + 1 : text.index("*** END OF")]
    # Skip the table of contents so every prompt is prose.
    body = body[body.index("\u201cWell, Prince, so Genoa") :]

    snapshot = snapshot_download(
        model, revision=revision, allow_patterns=["tokenizer.json", "tokenizer_config.json", "encoding/*.py"]
    )
    tok = PreTrainedTokenizerFast.from_pretrained(snapshot)
    spec = importlib.util.spec_from_file_location("encoding_dsv4", f"{snapshot}/encoding/encoding_dsv4.py")
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)

    def templated(content: str) -> int:
        prompt = encoding.encode_messages([{"role": "user", "content": content}], thinking_mode="chat")
        return len(tok.encode(prompt, add_special_tokens=False))

    ids = tok.encode(body, add_special_tokens=False)
    nonce = NONCE_TEMPLATE.format(uuid.uuid4().hex)
    manifest = {
        "source": SOURCE,
        "source_sha256": SOURCE_SHA256,
        "model": model,
        "revision": revision,
        "body_tokens": len(ids),
        "nonce_template": NONCE_TEMPLATE,
        "templated_tokens": {},
    }
    for name, target in TARGETS.items():
        n = target - templated(nonce + INSTRUCTION)
        # Tokens can merge across the cut, so the count is corrected until it lands on the target.
        for _ in range(5):
            if n > len(ids):
                sys.exit(f"source has {len(ids)} tokens, fewer than {n}")
            content = INSTRUCTION + tok.decode(ids[:n])
            got = templated(nonce + content)
            if got == target:
                break
            n += target - got
        (out / f"{name}.txt").write_text(content)
        manifest["templated_tokens"][name] = got
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
