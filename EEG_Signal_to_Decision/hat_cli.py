"""Repository launcher; keep interpreter setup out of shell-quoted Python code."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "EEG_Realtime"))
import eeg_realtime  # Activates matching project-local wheels when present.
import numpy
import scipy
import sklearn
import joblib
sys.path.insert(0, str(root / "src"))

if __name__ == "__main__" and sys.argv[1:] != ["--check-runtime"]:
    sys.argv[0] = "bci.hat_pipeline"
    runpy.run_module("bci.hat_pipeline", run_name="__main__")
