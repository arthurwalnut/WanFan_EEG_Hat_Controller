"""Standalone loopback MI application, with no legacy or visual-workflow imports."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .sources import discover_devices

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 65536
STATIC_FILES = {"/": "mi.html", "/mi": "mi.html", "/mi.html": "mi.html",
                "/mi.js": "mi.js", "/mi.css": "mi.css",
                "/robot-ui.js": "robot-ui.js", "/robot-ui.css": "robot-ui.css"}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


class MIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, engine):
        if address[0] != "127.0.0.1":
            raise ValueError("本应用仅允许监听 127.0.0.1。")
        self.engine = engine
        self.csrf_token = secrets.token_urlsafe(32)
        self.discovery_lock = threading.Lock()
        self.workflow_lock = threading.RLock()
        super().__init__(address, MIHandler)

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def server_close(self):
        try:
            self.engine.close()
        finally:
            super().server_close()


class MIHandler(BaseHTTPRequestHandler):
    server: MIServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        if args and str(args[1] if len(args) > 1 else "") not in ("200", "304"):
            super().log_message(format, *args)

    def _headers(self, code, content_type, length):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()

    def _json(self, value, code=200):
        payload = json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self._headers(code, "application/json; charset=utf-8", len(payload))
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _host_ok(self):
        port = self.server.server_address[1]
        values = self.headers.get_all("Host", [])
        return len(values) == 1 and values[0] in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _reject_post(self, message, code):
        # Draining a small already-submitted body avoids a TCP reset hiding the
        # error on Windows. The total drain deadline remains bounded.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if 0 < length <= MAX_BODY and not self.headers.get("Transfer-Encoding"):
            old_timeout = self.connection.gettimeout()
            deadline = time.monotonic() + .5
            try:
                while length and time.monotonic() < deadline:
                    self.connection.settimeout(max(.001, deadline - time.monotonic()))
                    block = self.rfile.read1(min(length, 8192))
                    if not block:
                        break
                    length -= len(block)
            except OSError:
                pass
            finally:
                self.connection.settimeout(old_timeout)
        self._json({"error": message}, code)

    def do_GET(self):
        if not self._host_ok():
            self._json({"error": "Host 不合法。"}, 403)
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                self._json({"service": "eeg-mi-robot", "api_version": 1,
                            "host": "127.0.0.1", "port": self.server.server_address[1],
                            "pid": os.getpid(), "workflows": ["mi"]})
            elif path == "/api/mi/clock":
                self._json({"server_time": time.monotonic(),
                            "clock_resolution_s": time.get_clock_info("monotonic").resolution})
            elif path == "/api/mi/bootstrap":
                self._json({"csrf_token": self.server.csrf_token, "server_time": time.monotonic(),
                            "sessions": self.server.engine.sessions(), "models": self.server.engine.model_files(),
                            "channels": ["O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2"],
                            "sample_rate": 250, "version": "2.0.0", "workbenches": ["mi"]})
            elif path == "/api/mi/state":
                self._json(self.server.engine.snapshot())
            elif path == "/api/mi/predictions":
                after = int(parse_qs(urlsplit(self.path).query).get("after", ["0"])[0])
                self._json(self.server.engine.prediction_records(after))
            elif path in STATIC_FILES:
                filename = STATIC_FILES[path]
                payload = (STATIC / filename).read_bytes()
                kind = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[Path(filename).suffix]
                self._headers(200, kind + "; charset=utf-8", len(payload))
                self.wfile.write(payload)
            else:
                self._json({"error": "资源不存在。"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except (ValueError, TypeError) as exc:
            self._json({"error": str(exc)}, 400)
        except FileNotFoundError:
            self._json({"error": "文件不存在，请完整解压或刷新列表。"}, 404)
        except Exception as exc:
            self._json({"error": f"操作失败：{exc}"}, 500)

    def do_POST(self):
        self.close_connection = True
        port = self.server.server_address[1]
        origins = self.headers.get_all("Origin", [])
        tokens = self.headers.get_all("X-EEG-Token", [])
        if (not self._host_ok() or len(origins) > 1 or
                (origins and origins[0] not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")) or
                len(tokens) != 1 or not tokens[0].isascii() or
                not secrets.compare_digest(tokens[0], self.server.csrf_token)):
            self._reject_post("本地请求校验失败，请刷新本应用页面。", 403)
            return
        if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
            self._reject_post("请求需要唯一 Content-Length，不能使用分块编码。", 400)
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._reject_post("请求必须是 application/json。", 415)
            return
        try:
            length = int(self.headers["Content-Length"])
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            self._json({"error": "请求长度无效或超过 64 KB。"}, 413)
            return
        try:
            def invalid_constant(value):
                raise ValueError("JSON 不接受 NaN/Infinity。")
            deadline = time.monotonic() + 3.0
            chunks = []
            remaining = length
            old_timeout = self.connection.gettimeout()
            try:
                while remaining:
                    self.connection.settimeout(max(.001, deadline - time.monotonic()))
                    block = self.rfile.read1(min(remaining, 8192))
                    if not block:
                        raise ValueError("请求正文不完整。")
                    chunks.append(block)
                    remaining -= len(block)
                    if remaining and time.monotonic() >= deadline:
                        raise TimeoutError("请求正文接收超时。")
            finally:
                self.connection.settimeout(old_timeout)
            body = json.loads(b"".join(chunks).decode("utf-8"), parse_constant=invalid_constant)
            if not isinstance(body, dict):
                raise ValueError("请求必须为 JSON 对象。")
            self._json(self._dispatch(urlsplit(self.path).path, body))
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except TimeoutError:
            self._json({"error": "请求正文接收超时。"}, 408)
        except FileNotFoundError:
            self._json({"error": "文件已不存在，请刷新列表。"}, 404)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, 409)
        except Exception as exc:
            self._json({"error": f"操作失败：{exc}"}, 500)

    def _dispatch(self, path, body):
        if path == "/api/discover":
            if not self.server.discovery_lock.acquire(blocking=False):
                raise RuntimeError("正在发现设备，请稍后重试。")
            try:
                return {"devices": discover_devices(timeout=2.0, device_ip=body.get("device_ip"))}
            finally:
                self.server.discovery_lock.release()
        engine = self.server.engine
        actions = {
            "/api/mi/start": lambda: engine.start(body),
            "/api/mi/stop": engine.stop,
            "/api/mi/record/start": lambda: engine.start_recording(body),
            "/api/mi/record/stop": lambda: engine.stop_recording(body),
            "/api/mi/cue": lambda: engine.acknowledge_cue(body),
            "/api/mi/display": lambda: engine.acknowledge_display(body),
            "/api/mi/replay/load": lambda: engine.load_replay(body.get("id")),
            "/api/mi/replay/control": lambda: engine.replay_control(body),
            "/api/mi/train": lambda: engine.train(body),
            "/api/mi/model/load": lambda: engine.load_model(body.get("id")),
            "/api/mi/robot/configure": lambda: engine.robot_configure(body),
            "/api/mi/robot/heartbeat": lambda: engine.robot_heartbeat(body),
            "/api/mi/robot/test": engine.robot_test,
            "/api/mi/robot/disarm": lambda: engine.robot_disarm(body),
        }
        if path not in actions:
            raise ValueError("未知 MI 操作。")
        with self.server.workflow_lock:
            return actions[path]()


def main(argv=None):
    parser = argparse.ArgumentParser(description="MI 采集、训练与机械臂本地实验台")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port 应在 1024～65535 之间")
    from .mi_engine import MIEngine
    engine = MIEngine(ROOT)
    server = None
    try:
        server = MIServer(("127.0.0.1", args.port), engine)
        print(f"MI ready: http://127.0.0.1:{args.port}/ ; Ctrl+C exits. No device connected.", flush=True)
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server is None:
            engine.close()
        else:
            server.server_close()


if __name__ == "__main__":
    main()
