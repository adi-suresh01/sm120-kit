"""Measures prefill throughput on the live server with the issue's workloads, through the OpenAI chat API.

Order: an arithmetic probe (is the output sane at all), one 8K warmup, then BENCH_REPS (default 2) reps of each
workload in BENCH_WORKLOADS (default w1,w2):
  w1  the reporter's: 9 concurrent ~90K-token prompts, 61 output tokens each
  w2  hermia's: one ~238K-token prompt, 40 output tokens
A row's prefill_tok_s is the server's usage.prompt_tokens summed over the requests, divided by the seconds from the
first request sent to the last response received. Requests stream, so each one's time to first token comes along.
Every prompt starts with a fresh nonce: W1's prompts share one text, and a prefix cache would otherwise serve
eight of them from the first.
Writes <out_dir>/prefill_bench.jsonl and exits 1 when any request fails, so the harness reports POST_FAIL.
Usage: prefill_bench.py <port> <model> <engine> <variant> <out_dir> <prompt_dir>
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKLOADS = {"warmup": (1, 16), "w1": (9, 61), "w2": (1, 40)}
# The last W1 request waits behind eight 90K prefills: about 400 s at 2K tok/s, so the read timeout leaves 4x that.
TIMEOUT_S = 1800
PROBE = "What is 17 * 23? Answer with just the number."


def chat(port: int, model: str, content: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    sent = time.monotonic()
    first, usage, finish, error, pieces = None, None, None, None, []
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    error = json.dumps(chunk["error"])[:500]
                    break
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        first = first or time.monotonic()
                        pieces.append(piece)
                    finish = choice.get("finish_reason") or finish
    except urllib.error.HTTPError as e:
        error = f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}"
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
        error = repr(e)
    done = time.monotonic()
    usage = usage or {}
    if error is None and not usage.get("prompt_tokens"):
        error = "the stream ended without usage.prompt_tokens"
    return {
        "ok": error is None,
        "error": error,
        "sent": sent,
        "done": done,
        "ttft_s": round(first - sent, 2) if first else None,
        "e2e_s": round(done - sent, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        # Only servers that report cache hits fill this; for the rest the server log says (README.md).
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": finish,
        "text": "".join(pieces),
    }


def run(port: int, model: str, contents: list[str], max_tokens: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(contents)) as pool:
        return list(pool.map(lambda c: chat(port, model, c, max_tokens), contents))


def main() -> None:
    port, model, engine, variant = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    out, prompt_dir = Path(sys.argv[5]), Path(sys.argv[6])
    manifest = json.loads((prompt_dir / "manifest.json").read_text())
    reps = int(os.environ.get("BENCH_REPS", "2"))
    names = [n for n in os.environ.get("BENCH_WORKLOADS", "w1,w2").split(",") if n]
    plan = [("warmup", 1)] + [(name, rep) for name in names for rep in range(1, reps + 1)]
    texts = {name: (prompt_dir / f"{name}.txt").read_text() for name, _ in plan}

    with open(out / "prefill_bench.jsonl", "a") as f:
        probe = chat(port, model, PROBE, 512)
        row = {"engine": engine, "variant": variant, "workload": "probe-17x23", "rep": 0, "ok": probe["ok"],
               "error": probe["error"], "correct": "391" in probe["text"], "sample": probe["text"][:200]}
        f.write(json.dumps(row) + "\n")
        f.flush()
        print("probe", "ok" if probe["ok"] else "FAILED", "correct" if row["correct"] else "WRONG", flush=True)
        if not probe["ok"]:
            sys.exit(1)

        for name, rep in plan:
            n, max_tokens = WORKLOADS[name]
            contents = [manifest["nonce_template"].format(uuid.uuid4().hex) + texts[name] for _ in range(n)]
            results = run(port, model, contents, max_tokens)
            ok = all(r["ok"] for r in results)
            wall = max(r["done"] for r in results) - min(r["sent"] for r in results)
            prompt_tokens = sum(r["prompt_tokens"] for r in results) if ok else None
            row = {
                "engine": engine,
                "variant": variant,
                "workload": name,
                "rep": rep,
                "requests": n,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": sum(r["completion_tokens"] or 0 for r in results) if ok else None,
                "built_tokens": manifest["templated_tokens"][name] * n,
                "wall_s": round(wall, 2),
                "prefill_tok_s": round(prompt_tokens / wall, 1) if ok else None,
                "ttft_max_s": max((r["ttft_s"] for r in results if r["ttft_s"] is not None), default=None),
                "ok": ok,
                "error": next((r["error"] for r in results if r["error"]), None),
                "sample": results[0]["text"][:200],
                "per_request": [{k: r[k] for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "ttft_s",
                                                   "e2e_s", "finish_reason", "error")} for r in results],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(name, rep, "ok" if ok else "FAILED", row["prompt_tokens"], row["wall_s"], row["prefill_tok_s"],
                  flush=True)
            if not ok:
                # A crashed engine fails every later request too; one failure is the answer.
                sys.exit(1)


if __name__ == "__main__":
    main()
