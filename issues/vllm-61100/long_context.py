"""Fills the server with long real-prose prompts, the load a RAG app puts on it, before the short probes run again.

The reporter's garbage came from a production server, and their minimal 55-token request ran against that same
long-lived process; a clean fresh server does not rule out a fault that needs the KV cache filled and reused first.
LONG_CONTEXT is a comma list of <tokens>x<count>, sent as one concurrent batch per entry.
Usage: long_context.py <port> <model> <out_dir> <model_dir>
"""

import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

SOURCE = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
INSTRUCTION = "Below is a long excerpt from a novel. Summarize what happens in it in three sentences.\n\n"


def ask(port: int, model: str, content: str) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": 200, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return {"ok": False, "error": repr(e)}
    return {"ok": True, "seconds": round(time.monotonic() - t0, 1),
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "finish_reason": data["choices"][0]["finish_reason"],
            "text": (data["choices"][0]["message"]["content"] or "")[:600]}


def main() -> None:
    port, model, out, model_dir = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    with urllib.request.urlopen(SOURCE, timeout=120) as resp:
        text = resp.read().decode("utf-8-sig")
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    ids = tok.encode(text, add_special_tokens=False).ids
    failed = 0
    with open(out / "long_context.jsonl", "w") as f:
        for entry in os.environ["LONG_CONTEXT"].split(","):
            n, count = (int(x) for x in entry.split("x"))
            # Different slices per request, so prefix caching cannot serve one prompt for all of them.
            prompts = [INSTRUCTION + tok.decode(ids[i * 7919 : i * 7919 + n]) for i in range(count)]
            with concurrent.futures.ThreadPoolExecutor(count) as pool:
                for i, result in enumerate(pool.map(lambda p: ask(port, model, p), prompts)):
                    f.write(json.dumps({"tokens": n, "index": i, **result}) + "\n")
                    f.flush()
                    failed += not result["ok"]
                    print(n, i, "ok" if result["ok"] else "FAILED", result.get("seconds"),
                          repr(result.get("text", ""))[:100], flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
