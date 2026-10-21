"""UDP 5005 robot receiver. Defaults to dry-run; no action queue or retry.

CLENCH -> Elite, BLINK -> Piper. Importing this module opens no socket/SDK.
ACK completion reports script exit, not independent physical verification.
"""
import argparse
import ast
from collections import OrderedDict
from datetime import datetime, timezone
import importlib.util
import ipaddress
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

PROTOCOL = "eeg-mi-robot-v1"
COOLDOWN_SECONDS = 25.
SCRIPT_MAP = {"CLENCH": "eat_elite.py", "BLINK": "drink_piper.py"}
ROOT = Path(__file__).resolve().parent


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def ipv4(value):
    try:
        address = ipaddress.IPv4Address(value)
        if address.is_multicast or address.is_unspecified or str(address) == "255.255.255.255":
            raise ValueError("expected a unicast IPv4 address")
        return str(address)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_packet(packet):
    if len(packet) > 4096: raise ValueError("packet exceeds 4096 bytes")
    text = packet.decode("utf-8").strip()
    try: payload = json.loads(text)
    except ValueError: payload = {"cmd": text}
    if isinstance(payload, str): payload = {"cmd": payload}
    if not isinstance(payload, dict): raise ValueError("expected command JSON object")
    command = payload.get("cmd", payload.get("command", payload.get("decision", "")))
    if not isinstance(command, str): raise ValueError("invalid command")
    command = command.strip().upper()
    if command not in {*SCRIPT_MAP, "NEUTRAL", "TEST"}: raise ValueError("unknown command")
    if "protocol" in payload and payload["protocol"] != PROTOCOL: raise ValueError("protocol mismatch")
    session = payload.get("session_id")
    if session is not None and (not isinstance(session, str) or not 1 <= len(session) <= 128):
        raise ValueError("invalid session_id")
    seq = payload.get("seq")
    if seq is not None and (type(seq) is not int or not 0 <= seq <= 2 ** 53): raise ValueError("invalid seq")
    if session is not None and seq is None: raise ValueError("session packets require seq")
    request = payload.get("request_id")
    if request is not None and (not isinstance(request, str) or not 1 <= len(request) <= 128):
        raise ValueError("invalid request_id")
    return command, payload


class RobotReceiver:
    def __init__(self, *, sender_ip="127.0.0.1", execute=False, runner=None,
                 emit=None, clock=time.monotonic, wall_clock=time.time):
        self.sender_ip = ipv4(sender_ip)
        self.execute = bool(execute)
        self.runner = runner or (lambda command: None)
        self.emit = emit or (lambda ack, address: None)
        self.clock, self.wall_clock = clock, wall_clock
        self.lock = threading.RLock()
        self.active = False
        self.faulted = False
        self.busy_until = 0.
        self.worker = None
        self.high_sequences = OrderedDict()
        self.requests = OrderedDict()
        self.last_states = OrderedDict()

    def _ack(self, payload, command, status, reason="", **extra):
        return dict(type="robot_ack", protocol=PROTOCOL, session_id=payload.get("session_id"),
            request_id=payload.get("request_id"), seq=payload.get("seq"), cmd=command,
            status=status, mode="execute" if self.execute else "dry_run", reason=reason,
            completed=status in ("completed", "dry_run"),
            executed=True if status == "completed" else False if status in ("dry_run", "ignored", "test", "busy", "duplicate") else None,
            receiver_time=datetime.now(timezone.utc).isoformat(), **extra)

    def _send(self, ack, address):
        try: self.emit(ack, address)
        except OSError as exc: log(f"ACK send failed; action is never retried: {exc}")
        log(f"{ack['cmd']} seq={ack['seq']} status={ack['status']} mode={ack['mode']} {ack['reason']}")
        return ack

    @staticmethod
    def _bound(mapping, size=4096):
        while len(mapping) > size: mapping.popitem(last=False)

    def handle(self, packet, address):
        # Unauthorized senders receive no reply and can never invoke runner.
        if address[0] != self.sender_ip:
            log(f"ignored unconfigured sender {address[0]}")
            return None
        try: command, payload = parse_packet(packet)
        except (ValueError, UnicodeError) as exc:
            return self._send(self._ack({}, "", "ignored", str(exc)), address)
        with self.lock:
            session = payload.get("session_id", f"legacy-port-{address[1]}")
            key = (address[0], session)
            sequence = payload.get("seq")
            request_id = payload.get("request_id")
            state_key = (key, sequence) if sequence is not None else None
            duplicate = (sequence is not None and sequence <= self.high_sequences.get(key, -1))
            duplicate = duplicate or (request_id is not None and (address[0], session, request_id) in self.requests)
            if duplicate:
                prior = self.last_states.get(state_key, "unknown")
                return self._send(self._ack(payload, command, "duplicate", "already consumed; no action retry", original_status=prior), address)
            if sequence is not None and key not in self.high_sequences and len(self.high_sequences) >= 4096:
                return self._send(self._ack(payload, command, "ignored", "sender session limit reached; inspect and restart manually"), address)
            # Busy/expired/rejected packets are consumed too, never queued for later.
            if sequence is not None:
                self.high_sequences[key] = sequence; self.high_sequences.move_to_end(key)
            if request_id is not None:
                self.requests[(address[0], session, request_id)] = True
                self._bound(self.requests)
            def reply(status, reason="", **details):
                if state_key is not None:
                    self.last_states[state_key] = status; self._bound(self.last_states)
                return self._send(self._ack(payload, command, status, reason, **details), address)
            if "expires_at" in payload:
                try:
                    expires = datetime.fromisoformat(str(payload["expires_at"]).replace("Z", "+00:00"))
                    if expires.tzinfo is None: raise ValueError("expires_at needs UTC/timezone")
                    remaining = expires.timestamp() - self.wall_clock()
                    if not 0 < remaining <= 10: raise ValueError("expired request or unsynchronized clocks (maximum TTL 10s)")
                except (ValueError, TypeError, OverflowError) as exc: return reply("ignored", str(exc))
            if command == "TEST": return reply("test", "link check only; no robot action")
            if command == "NEUTRAL": return reply("ignored", "rest never triggers movement")
            if self.faulted: return reply("failed", "receiver locked after prior action failure; inspect robot and restart manually")
            now = self.clock()
            if self.active or now < self.busy_until:
                return reply("busy", "active action or 25-second cooldown; request discarded",
                             active=self.active, cooldown_remaining_s=max(0., self.busy_until - now))
            self.active = True
            self.busy_until = now + COOLDOWN_SECONDS
            ack = reply("accepted", "starting action" if self.execute else "dry-run accepted; SDK will not run")
            self.worker = threading.Thread(target=self._run, args=(command, reply), daemon=True, name="robot-action")
            try: self.worker.start()
            except RuntimeError as exc:
                self.active = False; self.faulted = True; self.worker = None
                return reply("failed", f"worker start failed: {exc}; receiver locked")
            return ack

    def _run(self, command, reply):
        try:
            if self.execute: self.runner(command)
            with self.lock:
                self.active = False
                reply("completed" if self.execute else "dry_run",
                      "action script exited successfully; no independent physical verification" if self.execute else "dry-run complete; no SDK/robot command sent")
        except Exception as exc:
            with self.lock:
                self.active = False; self.faulted = True
                reply("failed", f"{type(exc).__name__}: {exc}; receiver locked; no automatic retry")


def check_dependencies():
    checks = {}
    for command, name in SCRIPT_MAP.items():
        path = ROOT / name
        try: ast.parse(path.read_text(encoding="utf-8")); checks[name] = "syntax OK"
        except (OSError, SyntaxError) as exc: checks[name] = str(exc)
    for module in ("elite", "piper_sdk"):
        try: checks[module] = "available" if importlib.util.find_spec(module) is not None else "not installed"
        except (ImportError, ValueError) as exc: checks[module] = str(exc)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return all(checks[name] == "syntax OK" for name in SCRIPT_MAP.values()) and all(checks[m] == "available" for m in ("elite", "piper_sdk"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--sender-ip", type=ipv4, default="127.0.0.1", help="allowed EEG-PC IPv4; default loopback only")
    parser.add_argument("--execute", action="store_true", help="explicitly permit SDK action subprocesses")
    parser.add_argument("--elite-ip", type=ipv4, help="Elite IP override; otherwise ELITE_ROBOT_IP or original 192.168.137.200")
    parser.add_argument("--piper-can", help="optional activated SocketCAN name; otherwise PIPER_CAN_NAME or installed SDK default")
    parser.add_argument("--check", action="store_true", help="inspect dependencies/scripts only, without sockets or SDK initialization")
    args = parser.parse_args(argv)
    if args.check: return 0 if check_dependencies() else 1
    if not 1 <= args.port <= 65535: parser.error("port must be 1..65535")
    def runner(command):
        script = ROOT / SCRIPT_MAP[command]
        cmd = [sys.executable, str(script)]
        if command == "CLENCH" and args.elite_ip: cmd.extend(["--ip", args.elite_ip])
        if command == "BLINK" and args.piper_can: cmd.extend(["--can-name", args.piper_can])
        subprocess.run(cmd, check=True)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind, args.port))
        receiver = RobotReceiver(sender_ip=args.sender_ip, execute=args.execute, runner=runner,
            emit=lambda ack, address: sock.sendto(json.dumps(ack, ensure_ascii=True).encode("utf-8"), address))
        log(f"UDP {args.bind}:{args.port}; allowed sender={args.sender_ip}; mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
        log("No action queue/retry. Stopping this receiver does not stop an already moving robot; use the robot's physical emergency stop.")
        try:
            while True:
                packet, address = sock.recvfrom(4097)
                receiver.handle(packet, address)
        except KeyboardInterrupt:
            log(f"receiver stopped; action_active={receiver.active}. A running action may continue; physical emergency stop remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
