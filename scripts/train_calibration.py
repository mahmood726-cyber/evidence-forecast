"""Offline training driver for flip_forecaster_v1.pkl.

Usage (paths are examples; prefer env vars or CLI, see
evidence_forecast._aact_paths for AACT discovery):

    export AACT_PATH="$AACT_SNAPSHOT/studies.txt"
    python scripts/train_calibration.py \\
        --pairs path/to/pairs.csv \\
        --aact "$AACT_PATH" \\
        --out ./models
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast.calibration.features import build_features
from evidence_forecast.calibration.train import train_models
from evidence_forecast.calibration.validate import (
    validate_model, write_validation_report,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="MetaAudit version-pair CSV")
    ap.add_argument("--aact", required=True, help="AACT snapshot path")
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--cutoff", default="2023-01-01")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    features = build_features(Path(args.pairs), aact_path=Path(args.aact))
    print(f"built {len(features)} labelled rows; base rate {features['flip'].mean():.3f}")
    artifacts = train_models(features, models_dir=out, cutoff=args.cutoff)

    with open(artifacts.gbm_path, "rb") as f:
        bundle = pickle.load(f)
    report = validate_model(bundle, features, cutoff=args.cutoff, holdout_topic=None)
    write_validation_report(report, out / "validation_report_v1.json")
    print(f"AUC {report.auc:.3f} | Brier {report.brier:.3f} | n_test {report.n_test}")

    cardio_report = validate_model(bundle, features, cutoff=args.cutoff, holdout_topic="cardiology")
    write_validation_report(cardio_report, out / "validation_report_v1_cardiology.json")
    print(f"cardiology AUC {cardio_report.auc:.3f} | n {cardio_report.n_test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
