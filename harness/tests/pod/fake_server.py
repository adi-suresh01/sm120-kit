"""Stands in for an inference server in the serve_variant.sh tests.

Usage: fake_server.py ok|fatal|hang|smoke_fail|post_die [--fake-mode <mode>] [--fake-gsm8k <answers>]
                      [--served-model-name <name>] [--max-model-len <tokens>] [server flags...]
  --fake-mode replaces the first mode, so a variant's own flags can make the server crash
  ok          answers every request with 200
  fatal       prints a CUDA fault with the frame before it, then hangs like a rank waiting on its peer
  hang        never listens
  smoke_fail  answers /health, fails every chat request with 500
  post_die    answers every request, then exits 1 s after the first chat request, mid-evaluation
  --fake-gsm8k answers GSM8K chat requests (a prompt ending in "Answer:") from the test split in GSM8K_DIR:
    correct   with the reference answer
    garbage   with commas until max_tokens, the way vllm#61100's server answers
    error     with 500, every try
    flaky     with 500 on the first try of each question, then the reference answer
    A GSM8K request that is not greedy and 5-shot gets 400, so a run that scores also checks the prompt.
GET /v1/models lists --served-model-name (default fake/model) with --max-model-len (default 8192), as vLLM does.
FAKE_SERVER_RECORD=<file> appends one JSON line per start: the full argv, every CONTRACT_* variable, and what the
fake sglang and fakeinfer packages hold, so a test sees exactly what reached the server.
"""
import argparse
import http.server
import importlib
import json
import os
import sys
import threading
import time

parser = argparse.ArgumentParser()
modes = ["ok", "fatal", "hang", "smoke_fail", "post_die"]
parser.add_argument("mode", choices=modes)
parser.add_argument("--fake-mode", choices=modes)
parser.add_argument("--fake-gsm8k", choices=["correct", "garbage", "error", "flaky"])
parser.add_argument("--served-model-name", default="fake/model")
parser.add_argument("--max-model-len", type=int, default=8192)
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()
args.mode = args.fake_mode or args.mode


def installed(module, name):
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError:
        return None


if os.environ.get("FAKE_SERVER_RECORD"):
    # Only the test's own variables: the rest of the environment can hold credentials.
    record = {
        "argv": sys.argv,
        "env": {key: value for key, value in os.environ.items() if key.startswith("CONTRACT_")},
        "sglang": installed("sglang.core", "BEHAVIOR"),
        "fakeinfer": installed("fakeinfer.core", "KERNEL"),
    }
    with open(os.environ["FAKE_SERVER_RECORD"], "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

if args.mode == "fatal":
    print('  File "/opt/venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 331, in __call__')
    print("torch.AcceleratorError: CUDA error: an illegal memory access was encountered", flush=True)
if args.mode in ("fatal", "hang"):
    time.sleep(600)

death = threading.Timer(1.0, os._exit, [3])
answers = {}
if args.fake_gsm8k:
    with open(os.path.join(os.environ["GSM8K_DIR"], "test.jsonl"), encoding="utf-8") as f:
        answers = {row["question"]: row["answer"] for row in map(json.loads, f)}
tries = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def _reply(self, code, body=None):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True} if body is None else body).encode())

    def do_GET(self):
        if self.path == "/v1/models":
            card = {"id": args.served_model_name, "object": "model", "max_model_len": args.max_model_len}
            self._reply(200, {"object": "list", "data": [card]})
        else:
            self._reply(200)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if args.fake_gsm8k:
            request = json.loads(body)
            prompt = request["messages"][-1]["content"]
            if prompt.endswith("\nAnswer:"):
                self._gsm8k(request, prompt)
                return
        self._reply(500 if args.mode == "smoke_fail" else 200)
        if args.mode == "post_die" and not death.is_alive():
            death.start()

    def _gsm8k(self, request, prompt):
        if request.get("temperature") != 0 or prompt.count("Question: ") != 6:
            self._reply(400, {"error": "want greedy 5-shot"})
            return
        question = prompt.rsplit("Question: ", 1)[1].removesuffix("\nAnswer:")
        tries[question] = tries.get(question, 0) + 1
        if args.fake_gsm8k == "error" or (args.fake_gsm8k == "flaky" and tries[question] == 1):
            self._reply(500, {"error": "fake failure"})
            return
        if args.fake_gsm8k == "garbage":
            text, finish, tokens = "The,,,,,,,,,, and,,, ,,,,", "length", request["max_tokens"]
        else:
            text, finish, tokens = answers[question], "stop", len(answers[question].split())
        message = {"role": "assistant", "content": text}
        self._reply(200, {
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"completion_tokens": tokens},
        })

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
