"""Replays the reporter's exact request plus two plain English checks against the live server and saves the raw
output, so a garbage generation is on disk word for word.

Wrong answers are evidence, not failures: the script exits non-zero only when a request itself fails, and the
verdict's GSM8K score carries the pass/fail signal.
Usage: probe_requests.py <port> <model> <out_dir>
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Copied from the issue: Slovak system prompt, "What is 1+1? Answer only with a number.", expected "2".
REPORTER = {
    "messages": [
        {"role": "system", "content": "Odpovedaj v\u00fdlu\u010dne po slovensky, stru\u010dne a presne."},
        {"role": "user", "content": "Ko\u013eko je 1+1? Odpovedz iba \u010d\u00edslom."},
    ],
    "temperature": 0,
    "max_tokens": 256,
    "stream": False,
}
CASES = [
    ("reporter-slovak", "/v1/chat/completions", REPORTER, "2"),
    ("english-1plus1", "/v1/chat/completions",
     {"messages": [{"role": "user", "content": "What is 1+1? Answer with just the number."}],
      "temperature": 0, "max_tokens": 64}, "2"),
    ("completion-france", "/v1/completions",
     {"prompt": "The capital of France is", "temperature": 0, "max_tokens": 16}, "Paris"),
]


def main() -> None:
    port, model, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    failed = 0
    with open(out / "probe_requests.jsonl", "w") as f:
        for name, path, body, expect in CASES:
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                         data=json.dumps({"model": model, **body}).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.load(resp)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                f.write(json.dumps({"case": name, "ok": False, "error": repr(e)}) + "\n")
                failed += 1
                continue
            choice = data["choices"][0]
            text = choice["message"]["content"] if "message" in choice else choice["text"]
            text = text or ""
            row = {"case": name, "ok": True, "expect": expect, "pass": expect in text.strip()[:40],
                   "finish_reason": choice.get("finish_reason"), "usage": data.get("usage"), "text": text[:3000]}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(name, "PASS" if row["pass"] else "WRONG", row["finish_reason"], repr(text[:80]), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
