"""Stands in for an inference server in the serve_variant.sh tests.

Usage: fake_server.py ok|fatal|hang|smoke_fail|post_die [--fake-mode <mode>] [server flags...]
  --fake-mode replaces the first mode, so a variant's own flags can make the server crash
  ok          answers every request with 200
  fatal       prints a CUDA fault with the frame before it, then hangs like a rank waiting on its peer
  hang        never listens
  smoke_fail  answers /health, fails every chat request with 500
  post_die    answers every request, then exits 1 s after the first chat request, mid-evaluation
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


class Handler(http.server.BaseHTTPRequestHandler):
    def _reply(self, code):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_GET(self):
        self._reply(200)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._reply(500 if args.mode == "smoke_fail" else 200)
        if args.mode == "post_die" and not death.is_alive():
            death.start()

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
