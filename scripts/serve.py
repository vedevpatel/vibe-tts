"""A local page for the Expresso emotion fine-tune: type text, pick a style, hear it.

Usage: .venv/bin/python scripts/serve.py [--port 8765] [--no-open]     (or scripts/serve.sh)
Loads the [speak] checkpoint once (styled_voice.py: the same splitting, retries and --check as run.sh --style) and
serves scripts/page.html on http://127.0.0.1:<port>. Listens on this machine only. Every clip is also saved to
<output dir>/page/ with its text, style and log in clips.jsonl. Expresso speaker ex02's voice: noncommercial use only,
and a synthetic voice of a real person; do not present it as real (EXPRESSO.md).
"""
import argparse
import io
import json
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import soundfile as sf
from _common import output_dir  # before transformers: loads .env (HF_HOME)
from styled_voice import StyledVoice

PAGE = Path(__file__).with_name("page.html")
MAX_CHARS = 1000

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-open", action="store_true", help="do not open the page in the browser")
args = ap.parse_args()

print("loading the model...")
voice = StyledVoice()
lock = threading.Lock()   # one model, one generation at a time
clips_dir = output_dir() / "page"
clips_dir.mkdir(parents=True, exist_ok=True)
print(f"{voice.notice()}\ndevice={voice.device}, styles: {', '.join(voice.styles)}; clips saved to {clips_dir}")


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code, obj):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/info":
            self.send_json(200, {"styles": voice.styles, "notice": voice.notice(), "run": voice.run})
        elif self.path == "/history":
            log_file = clips_dir / "clips.jsonl"
            recs = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()] if log_file.is_file() else []
            recs = [r | {"url": f"/clips/{r['file']}"} for r in recs if (clips_dir / r["file"]).is_file()]
            self.send_json(200, recs[-30:][::-1])
        elif self.path.startswith("/clips/"):
            f = clips_dir / Path(self.path).name
            if f.suffix == ".wav" and f.is_file():
                self.send(200, f.read_bytes(), "audio/wav")
            else:
                self.send_json(404, {"error": "no such clip"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/speak":
            return self.send_json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self.send_json(400, {"error": "bad request"})
        text, style, check = str(req.get("text", "")).strip(), req.get("style"), bool(req.get("check"))
        if not text or len(text) > MAX_CHARS:
            return self.send_json(400, {"error": f"text must be 1-{MAX_CHARS} characters"})
        if style not in voice.styles:
            return self.send_json(400, {"error": f"style must be one of {voice.styles}"})
        log = []
        with lock:
            start = time.time()
            try:
                audio = voice.speak(text, style, check=check, log=log.append)
            except Exception as e:   # keep serving; show the reason on the page
                return self.send_json(500, {"error": f"{type(e).__name__}: {e}", "log": log})
            seconds = round(time.time() - start, 1)
        name = f"{datetime.now():%Y%m%d-%H%M%S}-{style}.wav"
        buf = io.BytesIO()
        sf.write(buf, audio, voice.sr, format="WAV", subtype="PCM_16")
        (clips_dir / name).write_bytes(buf.getvalue())
        rec = {"file": name, "text": text, "style": style, "check": check, "run": voice.run, "duration_s": round(len(audio) / voice.sr, 2),
               "generation_s": seconds, "log": log}
        with open(clips_dir / "clips.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name}: {rec['duration_s']}s audio in {seconds}s <- {text!r}" + "".join(f"\n  {m}" for m in log))
        self.send_json(200, rec | {"url": f"/clips/{name}"})

    def log_message(self, *a):   # the request lines are noise; generations are printed above
        pass


server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
url = f"http://127.0.0.1:{args.port}/"
print(f"open {url}  (Ctrl-C to stop)")
if not args.no_open:
    webbrowser.open(url)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nstopped")
