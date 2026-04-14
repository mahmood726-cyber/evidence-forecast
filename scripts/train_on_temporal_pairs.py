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
from evidence_forecast.calibration.train import train_models, split_temporal, _feature_cols
from evidence_forecast.calibration.validate import validate_model, write_validation_report


FEATURE_COLS = [
    "ci_width", "pi_width", "distance_from_null",
    "k", "tau2", "i2",
    "fragility_index", "egger_p", "trim_fill_delta", "benford_mad",
    "pipeline_trial_count", "pipeline_expected_n",
    "pipeline_sponsor_entropy", "pipeline_design_het", "pipeline_empty",
    # v1-intrinsic temporal features (pipeline-lite): derived per-pair from the
    # analysis's own study-year distribution. See extract_temporal_pairs.R.
    "v1_year_span", "v1_years_since_recent", "v1_annual_accrual",
]


def build_features_from_pairs(pairs_csv: Path) -> pd.DataFrame:
    labelled = label_flips(pairs_csv)
    raw = pd.read_csv(pairs_csv)
    aux_cols = ["ma_id", "v1_k", "v1_tau2", "v1_i2",
                "v1_year_span", "v1_years_since_recent", "v1_annual_accrual"]
    # If the enriched CSV is being used, also pull the real pipeline columns.
    enriched_cols = ["pipeline_trial_count", "pipeline_expected_n",
                     "pipeline_sponsor_entropy", "pipeline_design_het",
                     "pipeline_empty"]
    has_enriched = all(c in raw.columns for c in enriched_cols)
    if has_enriched:
        aux_cols = aux_cols + enriched_cols
    merged = labelled.merge(
        raw[aux_cols].drop_duplicates("ma_id"),
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
    # Pipeline features: use real AACT-derived values if present, else constants.
    if not has_enriched:
        merged["pipeline_trial_count"] = 0
        merged["pipeline_expected_n"] = 0
        merged["pipeline_sponsor_entropy"] = 0.0
        merged["pipeline_design_het"] = 0.0
        merged["pipeline_empty"] = True
    else:
        for col, default in [
            ("pipeline_trial_count", 0),
            ("pipeline_expected_n", 0),
            ("pipeline_sponsor_entropy", 0.0),
            ("pipeline_design_het", 0.0),
            ("pipeline_empty", True),
        ]:
            merged[col] = merged[col].fillna(default)

    # v1-intrinsic temporal features already in merged from aux_cols above.
    # Median-fill any missing values.
    for col in ["v1_year_span", "v1_years_since_recent", "v1_annual_accrual"]:
        merged[col] = merged[col].fillna(merged[col].median())

    keep = ["ma_id", "flip", "topic_area", "v1_date"] + FEATURE_COLS
    return merged[keep].copy()


def main() -> int:
    # Prefer the AACT-enriched pair CSV if it exists; fall back to the base.
    enriched = ROOT / "cache" / "temporal_cochrane_pairs_enriched.csv"
    base = ROOT / "tests" / "fixtures" / "temporal_cochrane_pairs_v0.csv"
    pairs_csv = enriched if enriched.exists() else base
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

    print("[3/5] Validating raw GBM on held-out (v1 >= 2015)")
    with open(artifacts.gbm_path, "rb") as f:
        bundle = pickle.load(f)
    report = validate_model(bundle, feats, cutoff="2015-01-01", holdout_topic=None)
    write_validation_report(report, models_dir / "validation_report_temporal.json")
    print(f"      raw AUC {report.auc:.3f}  Brier {report.brier:.3f}  "
          f"slope {report.calibration_slope:.2f}  n_test {report.n_test}")

    print("[4/5] Applying post-hoc Platt scaling (sigmoid calibration on train)")
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import clone
    train, _ = split_temporal(feats, cutoff="2015-01-01", holdout_topic=None)
    X_tr = train[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
    y_tr = train["flip"].astype(int).values
    # Wrap the un-fit pipeline in prefit calibration via CalibratedClassifierCV cv=5
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBClassifier
        base_clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=0, eval_metric="logloss", tree_method="hist",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        base_clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=0,
        )
    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", base_clf),
    ])
    cal = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=5)
    cal.fit(X_tr, y_tr)
    cal_bundle = {"pipeline": cal, "features": FEATURE_COLS, "schema_version": "1.0.0"}
    with open(artifacts.gbm_path, "wb") as f:
        pickle.dump(cal_bundle, f)
    cal_report = validate_model(cal_bundle, feats, cutoff="2015-01-01", holdout_topic=None)
    write_validation_report(cal_report, models_dir / "validation_report_temporal_calibrated.json")
    print(f"      calibrated AUC {cal_report.auc:.3f}  Brier {cal_report.brier:.3f}  "
          f"slope {cal_report.calibration_slope:.2f}  intercept {cal_report.calibration_intercept:.2f}")

    print("[5/6] Permutation sanity check (5 shuffles on calibrated model)")
    from sklearn.metrics import roc_auc_score
    _, test = split_temporal(feats, cutoff="2015-01-01", holdout_topic=None)
    X = test[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
    y = test["flip"].values
    p = cal_bundle["pipeline"].predict_proba(X)[:, 1]
    rng = np.random.default_rng(0)
    perm_aucs = [roc_auc_score(rng.permutation(y), p) for _ in range(5)]
    print(f"      permutation AUC mean {np.mean(perm_aucs):.3f} "
          f"(should be near 0.50)")

    print("[6/6] Ablation: pipeline-features family delta-AUC")
    from sklearn.metrics import brier_score_loss

    def _ablate(drop_cols: set, label: str) -> tuple[float, float]:
        keep = [c for c in FEATURE_COLS if c not in drop_cols]
        X_tr = train[keep].apply(pd.to_numeric, errors="coerce").values
        X_te = test[keep].apply(pd.to_numeric, errors="coerce").values
        cl = CalibratedClassifierCV(
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()),
                      ("clf", clone(base_clf))]),
            method="sigmoid", cv=5,
        )
        cl.fit(X_tr, y_tr)
        p_i = cl.predict_proba(X_te)[:, 1]
        auc_i = roc_auc_score(y, p_i)
        brier_i = brier_score_loss(y, p_i)
        print(f"      {label:30s} AUC {auc_i:.3f}  Brier {brier_i:.3f}")
        return auc_i, brier_i

    full_auc = cal_report.auc
    pipeline_cols = {"pipeline_trial_count", "pipeline_expected_n",
                     "pipeline_sponsor_entropy", "pipeline_design_het",
                     "pipeline_empty"}
    temporal_cols = {"v1_year_span", "v1_years_since_recent", "v1_annual_accrual"}
    print(f"      {'full (all features)':30s} AUC {full_auc:.3f}  Brier {cal_report.brier:.3f}")
    ab_pipe_auc, _ = _ablate(pipeline_cols, "ablated: no pipeline")
    ab_temp_auc, _ = _ablate(temporal_cols, "ablated: no temporal")
    ab_both_auc, _ = _ablate(pipeline_cols | temporal_cols, "ablated: neither family")
    print(f"      delta-AUC pipeline family:    {full_auc - ab_pipe_auc:+.3f}")
    print(f"      delta-AUC temporal family:    {full_auc - ab_temp_auc:+.3f}")
    print(f"      delta-AUC both families:      {full_auc - ab_both_auc:+.3f}")

    print("Done. Calibrated model at", artifacts.gbm_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
