"""GSM8K accuracy of a running OpenAI-compatible server, through /v1/chat/completions, for a package that ships no eval
client of its own (vLLM). quality.sh runs it on the server that just passed the smoke test.

5-shot from the first 5 train examples, greedy, over the first <questions> test questions, scored on the last number
in the reply against the number after "####" in the reference. The few-shot text is that of
issues/vllm-47990/eval_offline.py, but the protocol differs: it goes as one user message through the server's chat
template instead of as a raw prompt, it stops only on "Question:" (eval_offline.py also stops on three newlines), and
max_tokens follows max_model_len (see max_tokens) instead of a fixed 512.
Each reply goes to <out_dir>/gsm8k_responses.jsonl as it lands, so a run cut short keeps what it got. Then a line of
counts, and "Score: <accuracy>", the line verdict.sh also reads from sglang's run_eval. A request that timed out
scores as a wrong answer; one that failed fails the run, with no Score line (see main for why).
Usage: gsm8k_chat.py <port> <model> <questions> <out_dir>
  GSM8K_DIR (default /workspace/gsm8k) caches the data; GSM8K_MAX_TOKENS (default 4096) caps each reply;
  GSM8K_TIMEOUT_S replaces the per-request timeout that request_timeout derives from max_tokens.
"""

import concurrent.futures
import hashlib
import http.client
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


def attempt(call, retry_if):
    """(call()'s result, None, tries), or (None, the last error, tries) once retry_if(error) is false or the tries
    run out. Any error counts, a reply of the wrong shape included."""
    for tries in range(1, ATTEMPTS + 1):
        try:
            return call(), None, tries
        except Exception as e:
            if tries == ATTEMPTS or not retry_if(e):
                return None, e, tries
            time.sleep(tries)


def transient(error):
    """A refused or dropped connection, or a 5xx: what a second try can get past. Not a timeout: a request that ran
    out its time once would run it out again, and three of them would outlast serve_variant.sh's hour."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    if isinstance(error, urllib.error.URLError):
        error = error.reason
    return isinstance(error, (ConnectionError, http.client.HTTPException))


def timed_out(error):
    if isinstance(error, urllib.error.URLError):
        error = error.reason
    return isinstance(error, TimeoutError)


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
    if fresh:
        raw, error, _ = attempt(lambda: fetch(source), lambda _: True)
    else:
        try:
            with open(path, "rb") as f:
                raw, error = f.read(), None
        except OSError as e:
            error = e
    if error is not None:
        sys.exit(f"gsm8k: reading {source}: {describe(error)}")
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
    reply, error, _ = attempt(lambda: call(f"{base}/v1/models")["data"], transient)
    if error is not None:
        sys.exit(f"gsm8k: GET /v1/models: {describe(error)}")
    lengths = {card.get("id"): card.get("max_model_len") for card in reply}
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


def request_timeout(tokens):
    # Long enough for a reply of max_tokens at 4 tok/s, slower than a large model at full batch should decode, after
    # 10 minutes queued behind the other requests. Short enough that a reply that never ends costs a known slice of
    # serve_variant.sh's hour, not all of it.
    return float(os.environ.get("GSM8K_TIMEOUT_S", 600 + tokens / 4))


def last_number(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def answer(base, body, timeout):
    reply = call(f"{base}/v1/chat/completions", body, timeout)
    choice = reply["choices"][0]
    message = choice["message"]
    return {
        # A reasoning parser leaves content null when the budget runs out mid-thought. That scores as no answer,
        # never from the reasoning, and the reasoning is kept so the row shows which of the two it was.
        "response": message["content"] or "",
        # vLLM names it reasoning; older vLLM and sglang, reasoning_content.
        "reasoning": (message.get("reasoning") or message.get("reasoning_content") or "")[:2000],
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
    timeout = request_timeout(tokens)
    print(f"gsm8k: {count} questions, max_tokens {tokens} (max_model_len {model_len}), timeout {timeout:g} s",
          flush=True)
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
        started = time.monotonic()
        reply, error, tries = attempt(lambda: answer(base, body, timeout), transient)
        spent = {"attempts": tries, "seconds": round(time.monotonic() - started, 1)}
        if error is None:
            return {"outcome": "answered", **reply, **spent, "error": None}
        outcome = "timeout" if timed_out(error) else "failed"
        none = {"response": None, "reasoning": None, "finish_reason": None, "completion_tokens": None}
        return {"outcome": outcome, **none, **spent, "error": describe(error)}

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with (
        concurrent.futures.ThreadPoolExecutor(THREADS) as pool,
        open(os.path.join(out_dir, "gsm8k_responses.jsonl"), "w", encoding="utf-8") as f,
    ):
        futures = {pool.submit(ask, row): index for index, row in enumerate(test)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            reply = future.result()
            got = last_number(reply["response"] or "")
            want = last_number(test[index]["answer"].split("####")[-1])
            row = {
                "index": index,
                "question": test[index]["question"],
                **reply,
                "parsed": got,
                "reference": want,
                "correct": got is not None and want is not None and abs(got - want) < 1e-6,
            }
            # One line per reply as it lands, in landing order: a run the watchdog or the outer limit ends keeps
            # every reply it got, which for a model generating garbage is the evidence.
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)

    answered = [row for row in rows if row["outcome"] == "answered"]
    failed = [row for row in rows if row["outcome"] == "failed"]
    counts = {
        "answered": len(answered),
        "correct": sum(row["correct"] for row in rows),
        # Garbage reads as replies that ran to max_tokens, replies with no number, and requests that timed out. With
        # the stop on "Question:" a healthy reply ends on its own, so running to max_tokens is the tell even when the
        # last number before the garbage was right.
        "hit_max_tokens": sum(row["finish_reason"] == "length" for row in answered),
        "no_number": sum(row["parsed"] is None for row in answered),
        # Empty content is most often a reasoning model cut off mid-thought rather than garbage. It counts under
        # no_number too; this keeps the two apart.
        "empty_content": sum(row["response"] == "" for row in answered),
        "timed_out": sum(row["outcome"] == "timeout" for row in rows),
        "failed": len(failed),
        # Retries recover from a server that errs now and then, which is worth seeing, not hiding.
        "retried": sum(row["attempts"] > 1 for row in rows),
    }
    print("gsm8k: " + " ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    # A timed-out request scores as wrong and the run still scores. On a server that is still up, a reply that does
    # not finish in time is a model that does not stop, which is what garbage does; failing the run would turn that
    # into a run with no score. A server that died, faulted, or went quiet is for the watchdog to report. A failed
    # request is different: the server refused or dropped it, so a score would count the server's errors as the
    # model's answers.
    if failed:
        sys.exit(f"gsm8k: {len(failed)}/{count} requests failed against the server; first: {failed[0]['error']}")
    print(f"Score: {counts['correct'] / count:.3f}", flush=True)


if __name__ == "__main__":
    main()
