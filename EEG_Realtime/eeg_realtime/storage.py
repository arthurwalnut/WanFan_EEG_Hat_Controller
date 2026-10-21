"""Project-local recording and safe file selection. Never writes vendor files."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def unique_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}{suffix}"


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.data = self.root / "data"
        self.recordings = self.data / "recordings"
        self.models = self.data / "models"
        self.calibrations = self.data / "calibrations"
        for path in (self.recordings, self.models, self.calibrations):
            path.mkdir(parents=True, exist_ok=True)
        self.legacy = self.root.parent / "InBrain" / "recording"

    def replay_files(self) -> list[dict]:
        result = []
        for prefix, folder in (("vendor", self.legacy), ("local", self.recordings)):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.csv"), reverse=True):
                if path.is_symlink() or path.stat().st_size > 200 * 1024 * 1024:
                    continue
                fs, channels = None, 0
                try:
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        for _ in range(40):
                            line = handle.readline()
                            if not line:
                                break
                            if line.startswith("# SampleRate:"):
                                fs = int(float(line.split(":", 1)[1].strip()))
                            elif line.strip() and not line.startswith("#"):
                                names = next(csv.reader([line]))
                                channels = sum(n.upper().startswith("CH") and n[2:].isdigit() for n in names)
                                break
                    result.append({"id": f"{prefix}/{path.name}", "name": path.name,
                                   "sample_rate": fs, "channels": channels,
                                   "seconds": None, "bytes": path.stat().st_size})
                except (OSError, ValueError, UnicodeError):
                    continue
        return result

    def resolve_replay(self, identifier: str) -> Path:
        allowed = {item["id"] for item in self.replay_files()}
        if identifier not in allowed:
            raise ValueError("请从已有录制文件列表选择回放文件。")
        prefix, name = identifier.split("/", 1)
        base = self.legacy if prefix == "vendor" else self.recordings
        path = (base / name).resolve()
        if path.parent != base.resolve():
            raise ValueError("回放路径超出允许目录。")
        return path

    def model_files(self) -> list[dict]:
        return [{"name": p.name} for p in sorted(self.models.glob("*.json"), reverse=True)
                if not p.is_symlink() and p.stat().st_size < 2_000_000]

    def load_model(self, name: str) -> dict:
        if name not in {item["name"] for item in self.model_files()}:
            raise ValueError("请选择本项目保存的模型文件。")
        return json.loads((self.models / name).read_text(encoding="utf-8"))

    def save_json(self, kind: str, value: dict) -> str:
        folder = self.models if kind == "model" else self.calibrations
        name = unique_name(kind, ".json")
        path = folder / name
        # Unique name + exclusive creation; no existing model is overwritten.
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        return name


class Recorder:
    def __init__(self, store: Store, config: dict, source: str, start_sample: int):
        self.name = unique_name("EEG", ".csv")
        self.path = store.recordings / self.name
        self.handle = self.path.open("x", encoding="utf-8", newline="")
        self.events_path = self.path.with_suffix(".events.jsonl")
        try:
            self.events_handle = self.events_path.open("x", encoding="utf-8")
        except OSError:
            self.handle.close()
            raise
        self.start_sample = start_sample
        self.fs = config["sample_rate"]
        self.samples = 0
        self.closed = False
        self.pending_discontinuity = False
        metadata = {"SampleRate": self.fs, "Created": utc_now(), "Units": "uV",
                    "Source": source, "ChannelNames": json.dumps(config["channel_names"], ensure_ascii=False),
                    "Timebase": "sample-derived seconds; no device timestamp; gaps preserved when known",
                    "Markers": "software marker attached to next received sample; not hardware trigger"}
        try:
            for key, value in metadata.items():
                self.handle.write(f"# {key}: {value}\n")
            self.writer = csv.writer(self.handle)
            self.writer.writerow(["Time"] + [f"CH{i+1}" for i in range(config["channels"])] +
                                 ["Marker", "TaskLabel", "Trial", "Mode", "Sequence", "ReceiveTimeUTC", "Discontinuity"])
            self.event("recording_started", {"config": config, "source": source})
        except OSError:
            self.closed = True
            self.handle.close()
            self.events_handle.close()
            raise

    def event(self, kind: str, details: dict):
        if not self.closed:
            self.events_handle.write(json.dumps({"time": utc_now(), "kind": kind, **details},
                                               ensure_ascii=False, allow_nan=False) + "\n")
            self.events_handle.flush()

    def write(self, batch, start_sample: int, mode: str, annotations, marker: str = ""):
        timestamp = datetime.fromtimestamp(batch.received_at, timezone.utc).isoformat(timespec="milliseconds")
        rows = []
        for i, sample in enumerate(batch.samples):
            label, trial = annotations(start_sample + i)
            rows.append([f"{(start_sample + i - self.start_sample)/self.fs:.6f}"] +
                        [f"{float(v):.6f}" for v in sample] +
                        [marker if i == 0 else "", label, trial, mode,
                         batch.sequence if batch.sequence is not None else "", timestamp,
                         1 if i == 0 and self.pending_discontinuity else 0])
        self.writer.writerows(rows)
        self.samples += len(rows)
        self.handle.flush()
        self.pending_discontinuity = False

    def close(self):
        if not self.closed:
            try:
                self.event("recording_stopped", {"samples": self.samples})
            finally:
                self.closed = True
                try:
                    self.handle.close()
                finally:
                    self.events_handle.close()
