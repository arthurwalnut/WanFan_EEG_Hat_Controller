"""Loopback-only HTTP application; no cloud calls or device auto-connect."""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from .engine import Engine
from .sources import discover_devices
from .mi_test_job import handle_evaluation_get

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class EEGServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, engine):
        if address[0] != "127.0.0.1":
            raise ValueError("本应用仅允许监听 127.0.0.1。")
        self.engine = engine
        self.csrf_token = secrets.token_urlsafe(32)
        self.discovery_lock = threading.Lock()
        self.workflow_lock = threading.RLock()
        self.mi_engine = None
        self.cvep_engine = None
        self.ssvep_engine = None
        super().__init__(address, Handler)

    def get_mi_engine(self):
        with self.workflow_lock:
            if self.mi_engine is None:
                from .mi_engine import MIEngine
                self.mi_engine = MIEngine(self.engine.store.root)
            return self.mi_engine

    def get_cvep_engine(self):
        with self.workflow_lock:
            if self.cvep_engine is None:
                from .cvep_engine import CVEPEngine
                self.cvep_engine = CVEPEngine(self.engine.store.root)
            return self.cvep_engine

    def get_ssvep_engine(self):
        with self.workflow_lock:
            if self.ssvep_engine is None:
                from .ssvep_engine import SSVEPEngine
                self.ssvep_engine = SSVEPEngine(self.engine.store.root)
            return self.ssvep_engine

    @staticmethod
    def active(workflow):
        if workflow is None:
            return False
        # An error label can precede socket/thread cleanup. Reserve the device
        # until its acquisition worker has actually exited and recording closed.
        attributes = vars(workflow)
        worker = attributes.get("worker")
        evaluation = attributes.get("evaluation")
        return (workflow.status in ("connecting", "running", "replay") or
                (evaluation is not None and evaluation.active()) or
                attributes.get("recorder") is not None or
                (isinstance(worker, threading.Thread) and worker.is_alive()))

    def server_close(self):
        try:
            if self.ssvep_engine is not None:
                self.ssvep_engine.close()
        finally:
            try:
                if self.cvep_engine is not None:
                    self.cvep_engine.close()
            finally:
                try:
                    if self.mi_engine is not None:
                        self.mi_engine.close()
                finally:
                    super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: EEGServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Poll requests are intentionally quiet. The application shows acquisition events.
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
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _reject_post(self, message, code):
        # Closing with an unread request body can reset TCP on Windows and hide
        # the JSON error. Drain only a small declared body, with a total deadline;
        # malformed/oversized/chunked requests never trigger an unbounded read.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if 0 < length <= 65536 and not self.headers.get("Transfer-Encoding"):
            previous_timeout = self.connection.gettimeout()
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
                self.connection.settimeout(previous_timeout)
        self._json({"error": message}, code)

    def do_GET(self):
        if not self._host_ok():
            self._json({"error": "Host 不合法。"}, 403)
            return
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json({"service": "eeg-realtime", "api_version": 1,
                        "host": self.server.server_address[0], "port": self.server.server_address[1],
                        "pid": os.getpid(), "workflows": ["mi", "cvep", "ssvep"]})
        elif path in ("/api/cvep/clock", "/api/ssvep/clock"):
            # QPC is a separate high-resolution time domain. On Windows Python
            # 3.12 the old MI monotonic clock may have ~15.6 ms resolution.
            self._json({"server_time": time.perf_counter(), "clock_domain": "perf_counter",
                        "clock_resolution_s": time.get_clock_info("perf_counter").resolution})
        elif path.startswith(("/api/cvep/", "/api/ssvep/")):
            try:
                prefix = "/api/ssvep" if path.startswith("/api/ssvep/") else "/api/cvep"
                cvep = self.server.get_ssvep_engine() if prefix == "/api/ssvep" else self.server.get_cvep_engine()
                if path == prefix + "/bootstrap":
                    self._json({**cvep.bootstrap(), "csrf_token": self.server.csrf_token,
                                "server_time": time.perf_counter(), "clock_domain": "perf_counter"})
                elif path == prefix + "/state":
                    self._json(cvep.snapshot())
                elif path == prefix + "/predictions":
                    after = int(parse_qs(urlsplit(self.path).query).get("after", ["0"])[0])
                    self._json(cvep.prediction_records(after))
                else:
                    self._json({"error": "资源不存在。"}, 404)
            except (ValueError, TypeError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"视觉诱发流程无法初始化：{exc}"}, 500)
        elif path == "/api/mi/clock":
            # Small, lock-free clock response: waveform serialization and file
            # listing must not bias the browser's clock synchronization.
            self._json({"server_time": time.monotonic(),
                        "clock_resolution_s": time.get_clock_info("monotonic").resolution})
        elif path.startswith("/api/mi/"):
            try:
                engine = self.server.get_mi_engine()
                if path == "/api/mi/bootstrap":
                    self._json({"csrf_token": self.server.csrf_token, "server_time": time.monotonic(),
                                "sessions": engine.sessions(), "models": engine.model_files(),
                                "channels": ["O2", "O1", "P8", "P7", "C4", "C3", "FP1", "FP2"],
                                "sample_rate": 250, "version": "2.0.0"})
                elif path == "/api/mi/state":
                    self._json(engine.snapshot())
                elif handle_evaluation_get(self, engine.evaluation):
                    pass
                elif path == "/api/mi/predictions":
                    after = int(parse_qs(urlsplit(self.path).query).get("after", ["0"])[0])
                    self._json(engine.prediction_records(after))
                else:
                    self._json({"error": "资源不存在。"}, 404)
            except (ValueError, TypeError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"八通道流程无法初始化，请检查依赖：{exc}"}, 500)
        elif path == "/api/state":
            self._json(self.server.engine.snapshot())
        elif path == "/api/bootstrap":
            self._json({"csrf_token": self.server.csrf_token,
                        "files": self.server.engine.store.replay_files(),
                        "models": self.server.engine.store.model_files(),
                        "defaults": {"sample_rate": 250, "channels": 8},
                        "version": "1.0.0"})
        elif path in ("/", "/mi", "/mi.html", "/mi.js", "/mi-evaluation.js", "/mi.css", "/robot-ui.js", "/robot-ui.css", "/cvep", "/cvep.html", "/cvep.js", "/cvep.css", "/ssvep", "/ssvep.html", "/ssvep.js", "/ssvep.css", "/training-report.js", "/training-report.css", "/legacy", "/index.html", "/app.js", "/style.css"):
            filename = ("mi.html" if path in ("/", "/mi") else "cvep.html" if path == "/cvep"
                        else "ssvep.html" if path == "/ssvep"
                        else "index.html" if path == "/legacy" else path[1:])
            content = (STATIC / filename).read_bytes()
            content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[Path(filename).suffix]
            try:
                self._headers(200, content_type + "; charset=utf-8", len(content))
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self._json({"error": "资源不存在。"}, 404)

    def do_POST(self):
        # No CORS; cross-site forms cannot meet JSON + token + Origin requirements.
        self.close_connection = True
        origin = self.headers.get("Origin")
        port = self.server.server_address[1]
        allowed = (f"http://127.0.0.1:{port}", f"http://localhost:{port}")
        if (not self._host_ok() or (origin is not None and origin not in allowed) or
                not secrets.compare_digest(self.headers.get("X-EEG-Token", ""), self.server.csrf_token)):
            self._reject_post("本地请求校验失败，请刷新本应用页面。", 403)
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._reject_post("请求必须是 application/json。", 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= 65536:
            self._json({"error": "请求长度无效或超过 64 KB。"}, 413)
            return
        try:
            def invalid_constant(value):
                raise ValueError("JSON 不接受 NaN/Infinity。")
            body = json.loads(self.rfile.read(length).decode("utf-8"), parse_constant=invalid_constant)
            if not isinstance(body, dict):
                raise ValueError("请求必须为 JSON 对象。")
            value = self._dispatch(urlsplit(self.path).path, body)
            self._json(value)
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            error = {"error": str(exc)}
            if hasattr(exc, "cvep_start_diagnostics"):
                error["details"] = exc.cvep_start_diagnostics
            if hasattr(exc, "ssvep_start_diagnostics"):
                error["details"] = exc.ssvep_start_diagnostics
            self._json(error, 400)
        except FileNotFoundError:
            self._json({"error": "文件已不存在，请刷新列表。"}, 404)
        except RuntimeError as exc:
            error = {"error": str(exc)}
            if hasattr(exc, "cvep_start_diagnostics"):
                error["details"] = exc.cvep_start_diagnostics
            if hasattr(exc, "ssvep_start_diagnostics"):
                error["details"] = exc.ssvep_start_diagnostics
            self._json(error, 409)
        except Exception as exc:
            with self.server.engine.lock:
                self.server.engine._event("error", f"操作失败：{exc}")
            self._json({"error": f"操作失败：{exc}"}, 500)

    def _dispatch(self, path, body):
        engine = self.server.engine
        if path.startswith(("/api/cvep/", "/api/ssvep/")):
            is_ssvep = path.startswith("/api/ssvep/")
            cvep = self.server.get_ssvep_engine() if is_ssvep else self.server.get_cvep_engine()
            other_visual = self.server.cvep_engine if is_ssvep else self.server.ssvep_engine
            # Both visual workflows deliberately expose the same command schema.
            path = path.replace("/api/ssvep/", "/api/cvep/", 1) if is_ssvep else path
            actions = {
                "/api/cvep/start": lambda: cvep.start(body),
                "/api/cvep/stop": cvep.stop,
                "/api/cvep/session/start": lambda: cvep.start_session(body),
                "/api/cvep/session/stop": lambda: cvep.stop_session(body),
                "/api/cvep/frames": lambda: cvep.frames(body),
                "/api/cvep/train": lambda: cvep.train(body),
                "/api/cvep/model/load": lambda: cvep.load_model(body.get("id")),
                "/api/cvep/replay/load": lambda: cvep.load_replay(body.get("id")),
                "/api/cvep/replay/control": lambda: cvep.replay_control(body),
            }
            if path not in actions:
                raise ValueError("未知视觉诱发操作。")
            with self.server.workflow_lock:
                if path == "/api/cvep/stop" and vars(cvep).get("recorder") is not None:
                    raise RuntimeError("正在录制，请先在刺激页面结束实验；页面已关闭时请使用紧急停止。")
                if (path in ("/api/cvep/start", "/api/cvep/replay/load") or
                        (path == "/api/cvep/replay/control" and body.get("action") == "play")):
                    if self.server.active(engine) or self.server.active(self.server.mi_engine) or self.server.active(other_visual):
                        raise RuntimeError("其他工作台仍在采集或回放，请先停止，再使用当前视觉诱发工作台。")
                return actions[path]()
        if path.startswith("/api/mi/"):
            mi = self.server.get_mi_engine()
            actions = {
                "/api/mi/start": lambda: mi.start(body),
                "/api/mi/stop": mi.stop,
                "/api/mi/record/start": lambda: mi.start_recording(body),
                "/api/mi/record/stop": lambda: mi.stop_recording(body),
                "/api/mi/cue": lambda: mi.acknowledge_cue(body),
                "/api/mi/display": lambda: mi.acknowledge_display(body),
                "/api/mi/replay/load": lambda: mi.load_replay(body.get("id")),
                "/api/mi/replay/control": lambda: mi.replay_control(body),
                "/api/mi/train": lambda: mi.train(body),
                "/api/mi/evaluation/start": lambda: mi.start_evaluation(body),
                "/api/mi/evaluation/cancel": lambda: mi.evaluation.cancel(body.get("id")),
                "/api/mi/recordings/delete-preview": lambda: mi.preview_recording_deletion(body),
                "/api/mi/recordings/delete": lambda: mi.delete_recordings(body),
                "/api/mi/library/update": lambda: mi.update_library(body),
                "/api/mi/models/delete-preview": lambda: mi.preview_model_deletion(body),
                "/api/mi/models/delete": lambda: mi.delete_models(body),
                "/api/mi/model/load": lambda: mi.load_model(body.get("id")),
                "/api/mi/robot/configure": lambda: mi.robot_configure(body),
                "/api/mi/robot/heartbeat": lambda: mi.robot_heartbeat(body),
                "/api/mi/robot/test": mi.robot_test,
                "/api/mi/robot/disarm": lambda: mi.robot_disarm(body),
            }
            if path not in actions:
                raise ValueError("未知八通道操作。")
            # Both workspaces share the same physical UDP endpoint and device.
            with self.server.workflow_lock:
                if (path in ("/api/mi/start", "/api/mi/replay/load", "/api/mi/evaluation/start") or
                        (path == "/api/mi/replay/control" and body.get("action") == "play")):
                    if self.server.active(self.server.cvep_engine):
                        raise RuntimeError("c-VEP 工作台仍在采集或回放，请先在 /cvep 停止。")
                    if self.server.active(self.server.ssvep_engine):
                        raise RuntimeError("SSVEP工作台仍在采集或回放，请先在 /ssvep 停止。")
                    if self.server.active(engine):
                        raise RuntimeError("旧实验台仍在采集，请先在 /legacy 停止采集。")
                return actions[path]()
        if path == "/api/start":
            with self.server.workflow_lock:
                if self.server.active(self.server.cvep_engine):
                    raise RuntimeError("c-VEP 工作台仍在采集或回放，请先在 /cvep 停止。")
                if self.server.active(self.server.ssvep_engine):
                    raise RuntimeError("SSVEP工作台仍在采集或回放，请先在 /ssvep 停止。")
                if self.server.active(self.server.mi_engine):
                    raise RuntimeError("八通道工作台仍在采集或回放，请先停止。")
                return engine.start(body)
        actions = {
            "/api/start": lambda: engine.start(body),
            "/api/stop": engine.stop,
            "/api/demo": lambda: engine.set_demo(body.get("state")),
            "/api/mode": lambda: engine.set_mode(body.get("mode")),
            "/api/feedback": lambda: engine.set_feedback(body.get("target"), body.get("threshold")),
            "/api/record/start": engine.start_recording,
            "/api/record/stop": engine.stop_recording,
            "/api/marker": lambda: engine.marker(body.get("label")),
            "/api/calibration/start": lambda: engine.start_calibration(body),
            "/api/calibration/abort": engine.abort_calibration,
            "/api/calibration/train": engine.train,
            "/api/model/save": engine.save_model,
            "/api/model/load": lambda: engine.load_model(body.get("name")),
        }
        if path == "/api/discover":
            if not self.server.discovery_lock.acquire(blocking=False):
                raise RuntimeError("正在搜索设备，请稍等。")
            try:
                return {"devices": discover_devices(timeout=2.0, device_ip=body.get("device_ip"))}
            finally:
                self.server.discovery_lock.release()
        if path not in actions:
            raise ValueError("未知操作。")
        return actions[path]()


def main():
    parser = argparse.ArgumentParser(description="EEG Hat 本地实时反馈与个人校准分类")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="启动后打开本地浏览器")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port 应在 1024～65535 之间")
    engine = Engine(ROOT)
    try:
        server = EEGServer(("127.0.0.1", args.port), engine)
    except OSError as exc:
        parser.exit(1, f"端口 {args.port} 无法监听，请关闭旧实例或使用 --port。{exc}\n")
    url = f"http://127.0.0.1:{args.port}"
    print(f"EEG Realtime ready: {url}\nPress Ctrl+C to exit. No device is connected automatically.", flush=True)
    if args.open:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print("\nStopping acquisition...", flush=True)
    finally:
        engine.stop()
        server.server_close()


if __name__ == "__main__":
    main()
