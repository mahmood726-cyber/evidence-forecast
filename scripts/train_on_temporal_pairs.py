"""Train flip-forecaster on the real temporal Cochrane pairs.

Uses scripts/extract_temporal_pairs.R output (1,600+ real pairs). The
available v1-snapshot features are: ci_width, distance_from_null, k, tau2, i2.
Pipeline + forensics features are filled with zeros / means since we don't
have per-pair PICO definitions for AACT matching (yet). This produces an
"effect-geometry-only" flip-forecaster — a reduced but real model.

Usage:
    TRUTHCERT_HMAC_KEY=dev python scripts/train_on_temporal_pairs.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast.calibration.label_flips import label_flips
from evidence_forecast.calibration.train import train_models
from evidence_forecast.calibration.validate import validate_model, write_validation_report


FEATURE_COLS = [
    "ci_width", "pi_width", "distance_from_null",
    "k", "tau2", "i2",
    "fragility_index", "egger_p", "trim_fill_delta", "benford_mad",
    "pipeline_trial_count", "pipeline_expected_n",
    "pipeline_sponsor_entropy", "pipeline_design_het", "pipeline_empty",
]


def build_features_from_pairs(pairs_csv: Path) -> pd.DataFrame:
    labelled = label_flips(pairs_csv)
    raw = pd.read_csv(pairs_csv)
    merged = labelled.merge(
        raw[["ma_id", "v1_k", "v1_tau2", "v1_i2"]].drop_duplicates("ma_id"),
        on="ma_id", how="left", suffixes=("", "_dup"),
    )

    # Effect geometry from label_flips output
    merged["ci_width"] = merged["v1_ci_high"] - merged["v1_ci_low"]
    merged["pi_width"] = np.nan  # not available from point-CI pairs alone
    null_val = merged["scale"].map({"HR": 1.0, "OR": 1.0, "RR": 1.0,
                                    "RD": 0.0, "MD": 0.0, "SMD": 0.0})
    merged["distance_from_null"] = (merged["v1_point"] - null_val).abs()

    # v1 pool metrics from R extractor
    merged["k"] = merged["v1_k"]
    merged["tau2"] = merged["v1_tau2"]
    merged["i2"] = merged["v1_i2"]

    # Features not extractable per pair — fill with neutral constants
    merged["fragility_index"] = 0
    merged["egger_p"] = 0.5
    merged["trim_fill_delta"] = 0.0
    merged["benford_mad"] = 0.01
    merged["pipeline_trial_count"] = 0
    merged["pipeline_expected_n"] = 0
    merged["pipeline_sponsor_entropy"] = 0.0
    merged["pipeline_design_het"] = 0.0
    merged["pipeline_empty"] = True

    keep = ["ma_id", "flip", "topic_area", "v1_date"] + FEATURE_COLS
    return merged[keep].copy()


def main() -> int:
    pairs_csv = ROOT / "tests" / "fixtures" / "temporal_cochrane_pairs_v0.csv"
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"[1/4] Building features from {pairs_csv.name}")
    feats = build_features_from_pairs(pairs_csv)
    print(f"      rows {len(feats):,}  flip rate {feats['flip'].mean():.3f}")

    print("[2/4] Training GBM/RF/L1-LR (temporal split 2015-01-01, no cardio holdout)")
    # These are Cochrane reviews not tagged by condition — cardiology holdout disabled.
    # Temporal split: train v1_date < 2015, test v1_date >= 2015.
    artifacts = train_models(
        feats, models_dir=models_dir, cutoff="2015-01-01", holdout_topic=None,
    )
    print(f"      shipped: {artifacts.gbm_path.name}")

    print("[3/4] Validating on held-out (v1 >= 2015)")
    with open(artifacts.gbm_path, "rb") as f:
        bundle = pickle.load(f)
    report = validate_model(bundle, feats, cutoff="2015-01-01", holdout_topic=None)
    write_validation_report(report, models_dir / "validation_report_temporal.json")
    print(f"      AUC {report.auc:.3f}  Brier {report.brier:.3f}  "
          f"slope {report.calibration_slope:.2f}  n_test {report.n_test}")

    print("[4/4] Permutation sanity check (5 shuffles)")
    from sklearn.metrics import roc_auc_score
    from evidence_forecast.calibration.train import split_temporal
    _, test = split_temporal(feats, cutoff="2015-01-01", holdout_topic=None)
    X = test[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
    y = test["flip"].values
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    rng = np.random.default_rng(0)
    perm_aucs = [roc_auc_score(rng.permutation(y), p) for _ in range(5)]
    print(f"      permutation AUC mean {np.mean(perm_aucs):.3f} "
          f"(should be near 0.50)")

    print("Done. Model at", artifacts.gbm_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
