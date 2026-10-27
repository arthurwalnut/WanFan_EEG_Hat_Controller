"""Windows executable entry point; source workflows remain unchanged."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
import webbrowser


def location() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def existing_service(port: int) -> bool:
    # Ignore system HTTP proxies when probing this machine's loopback service.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=2) as reply:
            value = json.loads(reply.read(16384))
        return (value.get("service") == "eeg-realtime" and value.get("api_version") == 1
                and value.get("host") == "127.0.0.1" and value.get("port") == port
                and {"mi", "cvep", "ssvep"}.issubset(value.get("workflows", [])))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def runtime_check() -> dict:
    import importlib
    import importlib.metadata
    from eeg_realtime import server
    modules = ["eeg_realtime.mi_engine", "eeg_realtime.cvep_engine",
               "eeg_realtime.mi_gap_export",
               "eeg_realtime.ssvep_engine", "eeg_realtime.robot_bridge",
               "bci.hat_pipeline.model", "bci.cvep_pipeline.model",
               "bci.ssvep_pipeline.model"]
    imported = {name: str(importlib.import_module(name).__file__) for name in modules}
    assets = sorted(p.name for p in server.STATIC.iterdir() if p.is_file())
    for name in ("mi.html", "cvep.html", "ssvep.html", "index.html", "training-report.js"):
        if name not in assets:
            raise RuntimeError(f"缺少网页资源：{name}")
    return {"ok": True, "frozen": bool(getattr(sys, "frozen", False)),
            "python": sys.version.split()[0], "modules": imported, "assets": assets,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "narwhals")}}


def wait_for_saves(workflows) -> None:
    for workflow in workflows:
        if workflow is None:
            continue
        workers = [getattr(workflow, "training_thread", None)]
        workers += list(getattr(workflow, "validation_threads", []))
        for worker in workers:
            if worker is not None and worker.is_alive():
                print(f"等待后台任务保存完成：{worker.name}。请勿关闭窗口。", flush=True)
                worker.join()


def main() -> int:
    parser = argparse.ArgumentParser(description="EEG Hat：采集、录制、训练与在线识别")
    parser.add_argument("--port", type=int, default=8765, help="网页端口，默认 8765")
    parser.add_argument("--workspace", type=Path,
                        help="工作区根目录；数据写入其中的 EEG_Realtime/data，默认 EXE 所在目录")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--no-reuse", action="store_true", help="端口已用时退出，不打开已有 EEG 服务")
    parser.add_argument("--check", action="store_true", help="检查内置依赖和资源后退出，不连接设备")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口应在 1024–65535 之间")
    if args.check:
        print(json.dumps(runtime_check(), ensure_ascii=False, indent=2), flush=True)
        return 0
    workspace = (args.workspace or location()).expanduser().resolve()
    app_root = workspace / "EEG_Realtime"
    url = f"http://127.0.0.1:{args.port}/"
    if existing_service(args.port):
        if args.no_reuse or args.workspace is not None:
            raise RuntimeError(f"端口 {args.port} 已有 EEG 服务。请更换 --port 或结束原服务。")
        print(f"已有 EEG 服务：{url}\n本次仅打开已有服务，不启动第二个采集进程。\n"
              "数据目录由已有服务决定。如需启动此 EXE，请先结束原服务，或指定其他 --port。", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    from eeg_realtime.engine import Engine
    from eeg_realtime.server import EEGServer

    class ExclusiveEEGServer(EEGServer):
        allow_reuse_address = False

        def server_bind(self):
            # Windows SO_REUSEADDR can bind over an unrelated listener.
            if os.name == "nt":
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    # A one-file bundle unpacks resources to a temporary directory. Never use
    # that resource directory as the persistent recording/model destination.
    app_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=app_root):
        pass
    engine = Engine(app_root)
    server = None
    try:
        server = ExclusiveEEGServer(("127.0.0.1", args.port), engine)
        print(f"EEG Hat 已启动：{url}\n数据目录：{app_root / 'data'}\n"
              "请保留此窗口；先停止录制和发送，再按 Ctrl+C 退出。\n"
              "启动不会连接设备、播放刺激或发送机械臂动作。", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print("正在停止服务并保存已采集数据…", flush=True)
    finally:
        workflows = [engine]
        if server is not None:
            workflows += [server.mi_engine, server.cvep_engine, server.ssvep_engine]
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                engine.stop()
            finally:
                wait_for_saves(workflows)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        if len(sys.argv) == 1 and sys.stdin is not None and sys.stdin.isatty():
            input("启动失败。请保留以上错误信息；按回车退出。")
        raise SystemExit(1)
