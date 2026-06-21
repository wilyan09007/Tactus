#!/usr/bin/env python3
"""
Guitar digital-twin calibration server — localhost, stdlib only.

  python3 software/ai/vision/calib_serve.py

Open the printed URL in Chrome, hold your guitar at the prompted angles, click
the 4 neck corners on each frozen frame; the page overlays the fret grid (the
equal-temperament law) so you SEE it snap onto your real frets, then saves the
still + the clicked points to data/calib/<guitar_id>/. Offline, twin.py turns
those keyframes into the guitar's digital twin (no paper, no marker).
"""
import base64, json, os, socketserver, sys, webbrowser
import http.server
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "calib")
PORT = int(os.environ.get("TACTUS_CALIB_PORT", "8770"))


def safe(s):
    return "".join(c for c in str(s) if c.isalnum() or c in "-_.").strip() or "x"


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "X-Meta,Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204)

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html", "/calibrate.html"):
            try:
                with open(os.path.join(HERE, "calibrate.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, '{"error":"calibrate.html not found next to calib_serve.py"}')
        elif path == "/ping":
            self._send(200, '{"ok":true}')
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(n) if n else b""
        try:
            if path == "/save":
                meta = json.loads(base64.b64decode(self.headers.get("X-Meta", "")).decode())
                gid = safe(meta.get("guitar_id", "guitar"))
                d = os.path.join(DATA, gid)
                os.makedirs(d, exist_ok=True)
                stem = f'kf_{int(meta.get("idx", 0)):02d}_{safe(meta.get("label","pose"))}'
                fp = os.path.join(d, stem + ".png")
                with open(fp, "wb") as f:
                    f.write(body)
                meta["frame_file"] = os.path.relpath(fp, ROOT)
                with open(os.path.join(d, "meta.jsonl"), "a") as f:
                    f.write(json.dumps(meta) + "\n")
                rel = os.path.relpath(fp, ROOT)
                print(f"  saved keyframe  {rel}  ({len(body)//1024} KB)  "
                      f"pts={len(meta.get('points', []))}")
                self._send(200, json.dumps({"ok": True, "path": rel}))
            else:
                self._send(404, '{"error":"not found"}')
        except Exception as e:
            print(f"  !! save error: {e}", file=sys.stderr)
            self._send(500, json.dumps({"ok": False, "error": str(e)}))


def main():
    os.makedirs(DATA, exist_ok=True)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    port = PORT
    for _ in range(10):
        try:
            httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    else:
        print("Could not bind a port. Set TACTUS_CALIB_PORT and retry.")
        sys.exit(1)
    url = f"http://localhost:{port}"
    print("=" * 60)
    print("  TACTUS guitar digital-twin calibration")
    print(f"  >>> open this in CHROME:   {url}")
    print(f"  writing keyframes to:      {os.path.relpath(DATA, ROOT)}/<guitar_id>/")
    print("  leave this terminal open while you capture. Ctrl-C to stop.")
    print("=" * 60)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
