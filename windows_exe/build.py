"""Reproducible one-file build; no recordings, models or machine config bundled."""
from pathlib import Path
import importlib.metadata
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / "EEG_Realtime" / ".packages", ROOT / ".build" / "packager",
         ROOT / "EEG_Realtime", ROOT / "EEG_Signal_to_Decision" / "src"]
sys.path[:0] = [str(path) for path in paths if path.is_dir()]
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
os.environ["PYINSTALLER_CONFIG_DIR"] = str(ROOT / ".build" / "pyinstaller-cache")

if __name__ == "__main__":
    if os.name != "nt" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Windows x64 CPython 3.12 with the project's pinned dependencies.")
    expected = {}
    for line in (ROOT / "EEG_Realtime" / "requirements-tested.txt").read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==")
            expected[name] = version
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise SystemExit(f"{name}: expected {version}, found {actual}")
    import PyInstaller.__main__
    PyInstaller.__main__.run([
        str(ROOT / "windows_exe" / "EEG_Hat.spec"),
        "--distpath", str(ROOT / ".build" / "output"),
        "--workpath", str(ROOT / ".build" / "work"),
        "--noconfirm",
    ])
