"""Local EEG acquisition, feedback and individually calibrated classification."""

import json as _json
import sys as _sys
from pathlib import Path as _Path

# Optional project-local wheels, installed for one specific interpreter. Never
# expose cp312 extension modules to the user's Anaconda/cp311 environment.
_packages = _Path(__file__).resolve().parent.parent / ".packages"
_runtime = _packages / "runtime.json"
if _runtime.is_file():
    _info = _json.loads(_runtime.read_text(encoding="utf-8"))
    if (_info.get("python_version") == list(_sys.version_info[:2]) and
            _Path(_info.get("executable", "")).resolve() == _Path(_sys.executable).resolve()):
        _sys.path.insert(0, str(_packages))

__version__ = "2.0.0"
