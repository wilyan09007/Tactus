#!/usr/bin/env python3
"""
Tactus capture server — localhost, stdlib only (no pip install).

  python3 software/ai/capture/serve.py

Then open the printed URL in **Chrome**. The browser page records synced
webcam + mic per prompted run; THIS server writes the files straight into
data/raw/<session>/<player>/ (audio WAV + video webm + manifest.jsonl) so
nothing is downloaded by hand and nothing gets lost.

Why a server (not file://): getUserMedia + AudioWorklet need a secure context;
http://localhost counts as secure. The server also guarantees saving + lets the
page report disk paths/bytes back so you can SEE every run land.
"""
import base64, json, os, socketserver, sys, webbrowser
import http.server
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # repo root
DATA = os.path.join(ROOT, "data", "raw")
CALIB = os.path.join(ROOT, "data", "calib")          # unified: this server also serves the twin/calib page
PORT = int(os.environ.get("TACTUS_PORT", "8765"))


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
        self.send_header("Access-Control-Allow-Headers", "X-Meta,X-Kind,Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204)

    def log_message(self, *a):
        pass  # keep the terminal clean for the operator

    # ---- GET: the page + a status/QC endpoint ----
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html", "/capture.html"):
            try:
                with open(os.path.join(HERE, "capture.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, '{"error":"capture.html not found next to serve.py"}')
        elif path in ("/calibrate", "/calibrate.html", "/map"):
            try:
                with open(os.path.join(HERE, "..", "vision", "calibrate.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, '{"error":"calibrate.html not found in ../vision/"}')
        elif path == "/ping":
            self._send(200, '{"ok":true}')
        elif path == "/status":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            self._send(200, json.dumps(self._status(q.get("session", [""])[0],
                                                    q.get("player", [""])[0])))
        else:
            self._send(404, '{"error":"not found"}')

    def _status(self, session, player):
        """Aggregate a session's manifest so the page can cross-check what's on disk."""
        out = {"runs": 0, "events": 0, "clipped": 0, "silent": 0, "coverage": {}}
        if not session or not player:
            return out
        mf = os.path.join(DATA, safe(session), safe(player), "manifest.jsonl")
        if not os.path.exists(mf):
            return out
        with open(mf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                out["runs"] += 1
                out["events"] += int(r.get("expected_note_count") or 0)
                a = r.get("audio") or {}
                out["clipped"] += 1 if a.get("clipped") else 0
                out["silent"] += 1 if a.get("silent") else 0
                key = f'{r.get("string","?")}|{r.get("intended_class","?")}'
                out["coverage"][key] = out["coverage"].get(key, 0) + 1
        return out

    # ---- POST: /save (one media blob) and /manifest (one row) ----
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(n) if n else b""
        try:
            if path == "/save":
                meta = json.loads(base64.b64decode(self.headers.get("X-Meta", "")).decode())
                kind = self.headers.get("X-Kind", "video")            # audio | video
                sub = "audio" if kind == "audio" else "video"
                ext = meta.get("ext", "wav" if kind == "audio" else "webm")
                d = os.path.join(DATA, safe(meta["session_id"]), safe(meta["player_id"]), sub)
                os.makedirs(d, exist_ok=True)
                fp = os.path.join(d, f'{safe(meta["run_id"])}.{ext}')
                with open(fp, "wb") as f:
                    f.write(body)
                rel = os.path.relpath(fp, ROOT)
                print(f"  saved {kind:5s}  {rel}  ({len(body)//1024} KB)")
                self._send(200, json.dumps({"ok": True, "path": rel, "bytes": len(body)}))
            elif path == "/calib-save":
                meta = json.loads(base64.b64decode(self.headers.get("X-Meta", "")).decode())
                gid = safe(meta.get("guitar_id", "guitar"))
                d = os.path.join(CALIB, gid)
                os.makedirs(d, exist_ok=True)
                stem = f'kf_{int(meta.get("idx", 0)):02d}_{safe(meta.get("label", "pose"))}'
                fp = os.path.join(d, stem + ".png")
                with open(fp, "wb") as f:
                    f.write(body)
                meta["frame_file"] = os.path.relpath(fp, ROOT)
                with open(os.path.join(d, "meta.jsonl"), "a") as f:
                    f.write(json.dumps(meta) + "\n")
                rel = os.path.relpath(fp, ROOT)
                print(f"  saved calib   {rel}")
                self._send(200, json.dumps({"ok": True, "path": rel}))
            elif path == "/manifest":
                row = json.loads(body.decode())
                d = os.path.join(DATA, safe(row["session_id"]), safe(row["player_id"]))
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "manifest.jsonl"), "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  + manifest row  run={row.get('run_id')}  "
                      f"class={row.get('intended_class')}  matched={row.get('matched_intent')}")
                self._send(200, json.dumps({"ok": True}))
            else:
                self._send(404, '{"error":"not found"}')
        except Exception as e:
            print(f"  !! save error: {e}", file=sys.stderr)
            self._send(500, json.dumps({"ok": False, "error": str(e)}))


def main():
    os.makedirs(DATA, exist_ok=True)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    port = PORT
    for attempt in range(10):
        try:
            httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    else:
        print("Could not bind a port 8765-8774. Set TACTUS_PORT and retry.")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print("=" * 60)
    print("  TACTUS capture server is running.")
    print(f"  >>> RECORD (open in CHROME): {url}")
    print(f"  >>> MAP THE GUITAR FIRST:    {url}/calibrate   (① one-time, ~5 min)")
    print(f"  writing recordings to:       {os.path.relpath(DATA, ROOT)}/<session>/<player>/")
    print(f"  writing calibration to:      {os.path.relpath(CALIB, ROOT)}/<guitar>/")
    print("  leave this terminal open while you record. Ctrl-C to stop.")
    print("=" * 60)
    # capture needs Chrome (getUserMedia + AudioWorklet); open it right on the URL
    # so you SEE it's connected the moment the server starts.
    try:
        if sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", "-a", "Google Chrome", url])
        else:
            webbrowser.open(url)
    except Exception:
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
