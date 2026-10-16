"""Stands in for an inference server in the serve_variant.sh tests.

Usage: fake_server.py ok|fatal|hang [server flags...]
  ok     answers every request with 200
  fatal  prints a CUDA fault with the frame before it, then hangs like a rank waiting on its peer
  hang   never listens
"""
import argparse
import http.server
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["ok", "fatal", "hang"])
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()

if args.mode == "fatal":
    print('  File "/opt/venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 331, in __call__')
    print("torch.AcceleratorError: CUDA error: an illegal memory access was encountered", flush=True)
if args.mode != "ok":
    time.sleep(600)


class Handler(http.server.BaseHTTPRequestHandler):
    def _ok(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_GET(self):
        self._ok()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._ok()

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
