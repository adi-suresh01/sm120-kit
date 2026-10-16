"""GSM8K accuracy of a running OpenAI-compatible server, through /v1/chat/completions, for a package that ships no eval
client of its own (vLLM). quality.sh runs it on the server that just passed the smoke test.

5-shot from the first 5 train examples, in the prompt format of issues/vllm-47990/eval_offline.py sent as one user
message, greedy, over the first <questions> test questions, scored on the last number in the reply against the number
after "####" in the reference. Prints "Score: <accuracy>", the line verdict.sh also reads from sglang's run_eval, and
writes every reply to <out_dir>/gsm8k_responses.jsonl. Exits non-zero, with no Score line, if any request still
failed after its retries.
Usage: gsm8k_chat.py <port> <model> <questions> <out_dir>
  GSM8K_DIR (default /workspace/gsm8k) caches the data; GSM8K_MAX_TOKENS (default 4096) caps each reply.
"""

import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

SHOTS = 5
# Enough client threads that the server's running-request cap, not the client, is the limit, as in sglang's runs.
THREADS = 64
ATTEMPTS = 3
# A reply that runs to max_tokens on a large model at full batch takes minutes. A server that hangs outright is left
# to serve_and_smoke.sh's stall check.
REQUEST_TIMEOUT_S = 1200
PROMPT_RESERVE = 2048
MIN_MAX_TOKENS = 256
COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{COMMIT}/grade_school_math/data/{{}}.jsonl"
# A truncated download, or another copy left in GSM8K_DIR by an issue's setup.sh, would change the score without
# failing anything. The tests point these at a small fixture.
SHA256 = {
    "train": os.environ.get("GSM8K_TRAIN_SHA256", "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
    "test": os.environ.get("GSM8K_TEST_SHA256", "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
}


def retried(call):
    """call(), tried ATTEMPTS times in all; the last try's error propagates."""
    for attempt in range(1, ATTEMPTS):
        # A dropped connection, a timeout, an HTTP error, and a reply of the wrong shape are all a failed try.
        try:
            return call()
        except Exception:
            time.sleep(attempt)
    return call()


def describe(error):
    if isinstance(error, urllib.error.HTTPError):
        # vLLM says why it refused a request in the body; the status line alone does not.
        return f"HTTP {error.code}: {error.read()[:300]!r}"
    return f"{type(error).__name__}: {error}"


def call(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def dataset(data_dir, split):
    """The split's rows, downloaded into data_dir once and checked against the pinned hash on every use."""
    path = os.path.join(data_dir, f"{split}.jsonl")
    fresh = not os.path.exists(path)
    source = URL.format(split) if fresh else path
    try:
        if fresh:
            raw = retried(lambda: fetch(source))
        else:
            with open(path, "rb") as f:
                raw = f.read()
    except Exception as e:
        sys.exit(f"gsm8k: reading {source}: {describe(e)}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256[split]:
        sys.exit(f"gsm8k: {source} has sha256 {digest}, want {SHA256[split]}")
    if fresh:
        os.makedirs(data_dir, exist_ok=True)
        # Renamed into place whole, so a run killed mid-write leaves no file that fails every later run.
        with open(f"{path}.part", "wb") as f:
            f.write(raw)
        os.replace(f"{path}.part", path)
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def max_tokens(base, model):
    try:
        cards = retried(lambda: call(f"{base}/v1/models"))["data"]
    except Exception as e:
        sys.exit(f"gsm8k: GET /v1/models: {describe(e)}")
    lengths = {card.get("id"): card.get("max_model_len") for card in cards}
    if model not in lengths:
        sys.exit(f"gsm8k: /v1/models does not list {model!r}, only {sorted(map(str, lengths))}")
    model_len = lengths[model]
    if not isinstance(model_len, int):
        sys.exit(f"gsm8k: /v1/models gives no max_model_len for {model!r}")
    cap = int(os.environ.get("GSM8K_MAX_TOKENS", "4096"))
    # vLLM answers 400 to a request whose prompt plus max_tokens exceeds max_model_len. The 5-shot prompt is under
    # 1k tokens even with the longest question, and the rest of the reserve covers a chat template's system prompt.
    # The cap matters as much: a model generating garbage never stops on its own, so without it every request would
    # run to the full context.
    tokens = min(cap, model_len - PROMPT_RESERVE)
    if tokens < MIN_MAX_TOKENS:
        # A worked answer cut off before its last line would score a healthy model as broken.
        sys.exit(
            f"gsm8k: max_tokens would be {tokens} (max_model_len {model_len} - {PROMPT_RESERVE}, at most {cap}), "
            f"under the {MIN_MAX_TOKENS} a worked answer needs"
        )
    return tokens, model_len


def last_number(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def answer(base, body):
    reply = call(f"{base}/v1/chat/completions", body, REQUEST_TIMEOUT_S)
    choice = reply["choices"][0]
    return {
        # A reasoning parser leaves content null when the budget runs out mid-thought; that scores as no answer.
        "response": choice["message"]["content"] or "",
        "finish_reason": choice["finish_reason"],
        "completion_tokens": (reply.get("usage") or {}).get("completion_tokens"),
    }


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: gsm8k_chat.py <port> <model> <questions> <out_dir>")
    port, model, count, out_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    data_dir = os.environ.get("GSM8K_DIR", "/workspace/gsm8k")
    train = dataset(data_dir, "train")[:SHOTS]
    test = dataset(data_dir, "test")
    if not 1 <= count <= len(test):
        sys.exit(f"gsm8k: asked for {count} questions; the test split has {len(test)}")
    test = test[:count]
    base = f"http://127.0.0.1:{port}"
    tokens, model_len = max_tokens(base, model)
    print(f"gsm8k: {count} questions, max_tokens {tokens} (max_model_len {model_len})", flush=True)
    shots = "".join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in train)

    def ask(row):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": f"{shots}Question: {row['question']}\nAnswer:"}],
            "temperature": 0,
            "max_tokens": tokens,
            # A model that goes on to write the next few-shot question would be scored on that question's numbers.
            "stop": ["Question:"],
        }
        try:
            return {**retried(lambda: answer(base, body)), "error": None}
        except Exception as e:
            return {"response": None, "finish_reason": None, "completion_tokens": None, "error": describe(e)}

    with concurrent.futures.ThreadPoolExecutor(THREADS) as pool:
        replies = list(pool.map(ask, test))
    rows = []
    for index, (row, reply) in enumerate(zip(test, replies)):
        got = last_number(reply["response"] or "")
        want = last_number(row["answer"].split("####")[-1])
        rows.append({
            "index": index,
            "question": row["question"],
            "response": reply["response"],
            "finish_reason": reply["finish_reason"],
            "completion_tokens": reply["completion_tokens"],
            "parsed": got,
            "reference": want,
            "correct": got is not None and want is not None and abs(got - want) < 1e-6,
            "error": reply["error"],
        })
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gsm8k_responses.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)

    correct = sum(row["correct"] for row in rows)
    failed = [row for row in rows if row["error"]]
    # Garbage output reads as replies with no number that ran to max_tokens.
    print(
        f"gsm8k: correct={correct} no_number={sum(row['parsed'] is None for row in rows)} "
        f"hit_max_tokens={sum(row['finish_reason'] == 'length' for row in rows)} failed={len(failed)}",
        flush=True,
    )
    if failed:
        # Scored as wrong answers, failed requests would pass a broken server as a weak model.
        sys.exit(f"gsm8k: {len(failed)}/{count} requests failed against the server; first: {failed[0]['error']}")
    print(f"Score: {correct / count:.3f}", flush=True)


if __name__ == "__main__":
    main()
