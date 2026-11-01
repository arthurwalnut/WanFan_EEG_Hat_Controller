"""Shared device ownership rules for the full and standalone controllers."""
import threading

from .impedance import ImpedanceEngine


def active(workflow):
    if workflow is None:
        return False
    worker = getattr(workflow, "worker", None)
    evaluation = getattr(workflow, "evaluation", None)
    return (workflow.status in ("connecting", "running", "replay") or
            getattr(workflow, "recorder", None) is not None or
            (evaluation is not None and evaluation.active()) or
            (isinstance(worker, threading.Thread) and worker.is_alive()))


def require_impedance_idle(server):
    if active(server.impedance_engine):
        raise RuntimeError("00 电极阻抗仍在运行，请先停止阻抗检查或点击切回普通 EEG。")


def dispatch(server, path, body, mi, others=()):
    """Caller holds workflow_lock throughout validation, stop and start."""
    impedance = server.impedance_engine
    if path == "/api/impedance/stop":
        return impedance.stop()
    if path not in ("/api/impedance/start", "/api/impedance/eeg"):
        raise ValueError("未知阻抗操作。")
    # Validate before stopping an existing session, including malformed IP input.
    options = body if path.endswith("/start") else {
        "source": "device", "device_ip": impedance.device_ip}
    ImpedanceEngine.validate_options(options)
    if any(active(other) for other in others):
        raise RuntimeError("其他工作台仍在采集或回放，请先停止，再检查阻抗。")
    # A recording may be between UI refreshes; check its authoritative state.
    with mi.lock:
        if mi.recorder is not None:
            raise RuntimeError("正在录制，请先结束并保存录制，再切换模式。")
        if mi.evaluation.active() or mi.training.get("status") == "running":
            raise RuntimeError("正在训练或测试模型，请完成后再切换模式。")
        if active(mi):
            can_switch = (path.endswith("/start") and body.get("switch_from_mi") is True
                          and mi.source_kind == "device"
                          and mi.status in ("running", "connecting")
                          and getattr(mi.source, "device_ip", None) == options["device_ip"].strip())
            if not can_switch:
                raise RuntimeError("当前采集或回放仍在运行，请先停止；同一设备可从普通 EEG 切入阻抗。")
    if path.endswith("/start"):
        if active(impedance):
            raise RuntimeError("阻抗检查已启动，请先停止后再更换设备。")
        # stop() also disarms robot output and joins the old acquisition worker.
        mi.stop()
        return impedance.start(options)
    if active(mi):
        raise RuntimeError("普通 EEG 工作台已在运行，请先停止当前会话。")
    impedance.stop()
    return mi.start({"source": "device", "device_ip": impedance.device_ip})
