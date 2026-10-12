"""Self-contained M3Sculptor UI server. Runs the workflow in-process, no backend."""
import argparse
import json
import os
import re
import signal
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from m3s.engine import Engine, SessionCapacityError

ROOT = Path(__file__).resolve().parent
FILES = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
         "/styles.css": ("styles.css", "text/css")}
SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
MAX_BODY = 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, mime="application/json"):
        raw = json.dumps(data, ensure_ascii=False).encode() if mime == "application/json" else data
        self.send_response(status)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        return

    def session_from(self, path):
        match = re.fullmatch(r"/api/session/([0-9a-f]+)(?:/(\w+))?", path)
        if not match or not SESSION_ID.fullmatch(match[1]):
            return None, None
        return self.server.engine.get(match[1]), match[2]

    def do_GET(self):
        if self.path == "/api/config":
            return self.reply(200, {"evaluation_mode": self.server.engine.evaluation_mode})
        if self.path.startswith("/api/session/"):
            session, action = self.session_from(self.path)
            if session is None or action not in (None, "state"):
                return self.reply(404, {"error": "This session does not exist or has expired. Please start a new conversation."})
            return self.reply(200, session.snapshot())
        if self.path not in FILES:
            return self.reply(404, {"error": "Not found"})
        name, mime = FILES[self.path]
        self.reply(200, (ROOT / name).read_bytes(), mime)

    def same_origin(self):
        """Allow any origin by default: the service is LAN-open and has no auth anyway.

        Set --strict-origin (or M3SCULPTOR_STRICT_ORIGIN=1) to require a same-host
        Origin. Port forwarding and tunnels change the port and hostname the browser
        sees, so strict mode is off unless explicitly requested.
        """
        if not getattr(self.server, "strict_origin", False):
            return True
        origin = self.headers.get("Origin")
        if not origin:
            return True
        if origin.rstrip("/") in getattr(self.server, "allowed_origins", ()):
            return True
        parts = urlsplit(origin)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return False
        host_name = self.headers.get("Host", "").partition(":")[0]
        return parts.hostname == host_name or {parts.hostname, host_name} <= LOOPBACK

    def do_POST(self):
        if not self.same_origin():
            print(f"[origin-rejected] Origin={self.headers.get('Origin')} Host={self.headers.get('Host')}", flush=True)
            return self.reply(403, {
                "error": "Cross-origin request rejected",
                "detail": f"Origin={self.headers.get('Origin')} Host={self.headers.get('Host')}."
                          "--strict-origin is enabled. Remove it when using port forwarding or a tunnel, "
                          "or allow this origin with --allow-origin <Origin shown above>."})
        try:
            payload = self.read_json()
        except ValueError as exc:
            return self.reply(400, {"error": str(exc)})
        if self.path == "/api/session":
            try:
                session = self.server.engine.create()
            except SessionCapacityError as exc:
                return self.reply(503, {"error": str(exc)})
            return self.reply(200, session.snapshot())
        session, action = self.session_from(self.path)
        if session is None or action != "message":
            return self.reply(404, {"error": "This session does not exist or has expired. Please start a new conversation."})
        try:
            turn = self.server.engine.submit(session, payload.get("text", ""))
        except ValueError as exc:
            return self.reply(400, {"error": str(exc)})
        except KeyError:
            return self.reply(404, {"error": "This session does not exist or has expired. Please start a new conversation."})
        if turn is None:
            return self.reply(409, {"error": "The previous turn is still running; the message was not sent again.", **session.snapshot()})
        self.reply(200, session.snapshot())

    def read_json(self):
        size = int(self.headers.get("Content-Length", "0") or 0)
        if size < 0 or size > MAX_BODY:
            raise ValueError("Request body is too large")
        if not size:
            return {}
        try:
            payload = json.loads(self.rfile.read(size))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Request body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload


def lan_addresses():
    """Best-effort list of this host's non-loopback IPv4 addresses."""
    found = []
    for family, _, _, _, address in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        if family == socket.AF_INET and not address[0].startswith("127."):
            found.append(address[0])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1: routes nowhere, sends nothing.
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    return sorted({ip for ip in found if not ip.startswith("127.")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default=os.getenv("M3SCULPTOR_HOST", "0.0.0.0"),
                        help="Bind address; default 0.0.0.0 allows LAN access. Use 127.0.0.1 for local access only.")
    parser.add_argument("--mode", choices=("normal", "test"), default=os.getenv("M3SCULPTOR_MODE", "normal"))
    parser.add_argument("--allow-origin", action="append", default=None,
                        help="Additional allowed Origin for --strict-origin; repeat this option or set comma-separated M3SCULPTOR_ALLOW_ORIGIN.")
    parser.add_argument("--strict-origin", action="store_true",
                        default=os.getenv("M3SCULPTOR_STRICT_ORIGIN", "") == "1",
                        help="Require matching Origin/Host names; port forwarding or tunnels may be rejected. Disabled by default.")
    args = parser.parse_args()
    origins = args.allow_origin or [o for o in os.getenv("M3SCULPTOR_ALLOW_ORIGIN", "").split(",") if o.strip()]
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.engine = Engine(evaluation_mode=args.mode)
    server.allowed_origins = tuple(o.strip().rstrip("/") for o in origins)
    server.strict_origin = args.strict_origin
    print(f"M3Sculptor started (mode={args.mode}), listening on {args.host}:{args.port}", flush=True)
    print(f"  Local access: http://127.0.0.1:{args.port}", flush=True)
    if args.host == "0.0.0.0":
        for address in lan_addresses():
            print(f"  LAN access: http://{address}:{args.port}", flush=True)
        print("  LAN access is enabled. The API has no authentication; use it only on trusted networks.", flush=True)
    if args.strict_origin:
        print("  --strict-origin is enabled; port forwarding or tunnel access may be rejected.", flush=True)

    def shutdown(*_):
        raise KeyboardInterrupt

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
