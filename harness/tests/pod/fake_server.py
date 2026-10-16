"""Stands in for an inference server in the serve_variant.sh tests.

Usage: fake_server.py ok|fatal|hang|smoke_fail|post_die [server flags...]
  ok          answers every request with 200
  fatal       prints a CUDA fault with the frame before it, then hangs like a rank waiting on its peer
  hang        never listens
  smoke_fail  answers /health, fails every chat request with 500
  post_die    answers every request, then exits 1 s after the first chat request, mid-evaluation
"""
import argparse
import http.server
import os
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["ok", "fatal", "hang", "smoke_fail", "post_die"])
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()

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
