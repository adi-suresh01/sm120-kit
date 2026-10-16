"""Sends the long prompts named in LONG_PROMPTS to the live server, one at a time, and records each outcome.

Order: a short arithmetic probe (is the output coherent at all), each long prompt in the order given (the crash
lands in the first 4092-token prefill chunk), then a 512-token generation at concurrency 1 for decode speed.
Writes <out_dir>/long_prompts.jsonl and exits 1 when any request fails, so the harness reports POST_FAIL.
Usage: long_prompts.py <port> <model> <out_dir> <prompt_dir>
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LONG_TIMEOUT_S = 900
SHORT_TIMEOUT_S = 300


def chat(port: int, model: str, content: str, max_tokens: int, timeout: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return {"ok": False, "error": repr(e), "seconds": round(time.monotonic() - t0, 2)}
    msg = data["choices"][0]["message"]
    usage = data.get("usage") or {}
    seconds = time.monotonic() - t0
    return {
        "ok": True,
        "seconds": round(seconds, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "tok_per_s": round(usage.get("completion_tokens", 0) / seconds, 1) if seconds else None,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "content": (msg.get("content") or "")[:2000],
        "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:2000],
    }


def main() -> None:
    port, model, out, prompt_dir = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    names = [n for n in os.environ.get("LONG_PROMPTS", "").split(",") if n]
    cases = [("probe-17x23", "What is 17 * 23? Answer with just the number.", 1024, SHORT_TIMEOUT_S)]
    cases += [(n, (prompt_dir / f"{n}.txt").read_text(), 256, LONG_TIMEOUT_S) for n in names]
    cases.append(("decode-512", "Write a 400-word story about a lighthouse keeper.", 512, SHORT_TIMEOUT_S))
    failed = 0
    with open(out / "long_prompts.jsonl", "a") as f:
        for name, content, max_tokens, timeout in cases:
            result = {"case": name, **chat(port, model, content, max_tokens, timeout)}
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(name, "ok" if result["ok"] else "FAILED", result.get("seconds"), result.get("prompt_tokens"),
                  result.get("tok_per_s"), flush=True)
            if not result["ok"]:
                failed += 1
                # A crashed engine fails every later request too; one failure is the answer.
                break
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
