"""Train only from explicit new EEG_Realtime MI session recordings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    if __package__ != "bci.hat_pipeline":
        raise SystemExit("Add EEG_Signal_to_Decision/src to PYTHONPATH and use python -m bci.hat_pipeline; do not use src.bci aliases for model serialization")
    parser = argparse.ArgumentParser(description="Eight-channel Hat decoder: new recordings only")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit and evaluate disjoint trial/session splits")
    train.add_argument("--recordings", nargs="+", required=True, help="New MI recording session paths")
    train.add_argument("--output", required=True, help="New .joblib model path; existing files are not overwritten")
    train.add_argument("--feature", choices=("bandpower", "csp", "combined"), default="bandpower")
    train.add_argument("--classifier", choices=("lda", "svm"), default="lda")
    args = parser.parse_args(argv)
    realtime = Path(__file__).resolve().parents[4] / "EEG_Realtime"
    if str(realtime) not in sys.path:
        sys.path.insert(0, str(realtime))
    from eeg_realtime.mi_recording import load_session
    from .model import train_model, save_model
    sessions = [load_session(Path(path)) for path in args.recordings]
    model = train_model(sessions, feature=args.feature, classifier=args.classifier)
    output = save_model(model, args.output)
    print(json.dumps({"model": str(output.resolve()), "source": model["source"], "metrics": model["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
