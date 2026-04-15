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
from evidence_forecast.calibration.train import (
    train_models, split_temporal, _feature_cols, FEATURE_COLS_V1,
)
from evidence_forecast.calibration.validate import validate_model, write_validation_report


FEATURE_COLS = FEATURE_COLS_V1


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
    # Distance from null on log scale for ratio scales, natural for differences.
    # Per advanced-stats.md: pool ratios on log scale; natural-scale distance
    # makes effect 2.0 (distance 1.0) and 0.5 (distance 0.5) asymmetric.
    scale_upper = merged["scale"].astype(str).str.upper()
    is_ratio = scale_upper.isin(["HR", "OR", "RR"])
    is_diff = scale_upper.isin(["RD", "MD", "SMD"])
    unknown = ~(is_ratio | is_diff)
    if unknown.any():
        bad = scale_upper[unknown].unique().tolist()
        raise ValueError(f"unknown effect scales in pairs: {bad}")
    point_safe = merged["v1_point"].where(merged["v1_point"] > 0, np.nan)
    distance = np.where(
        is_ratio,
        np.abs(np.log(point_safe)),  # log(null=1) = 0
        np.abs(merged["v1_point"]),  # null = 0
    )
    merged["distance_from_null"] = distance

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

    print("[2/6] Training GBM/RF/L1-LR (group-aware temporal split 2015-01-01, no cardio holdout)")
    # These are Cochrane reviews not tagged by condition — cardiology holdout disabled.
    # Group-aware split: each ma_id (review) goes wholly to train or test based
    # on its median v1_date crossing 2015-01-01. Eliminates within-review
    # cluster leakage (review-findings P0-5).
    artifacts = train_models(
        feats, models_dir=models_dir, cutoff="2015-01-01",
        holdout_topic=None, group_col="ma_id",
    )
    print(f"      shipped: {artifacts.gbm_path.name}")

    print("[3/6] Validating raw GBM on held-out (group-split, v1 >= 2015)")
    with open(artifacts.gbm_path, "rb") as f:
        bundle = pickle.load(f)
    report = validate_model(
        bundle, feats, cutoff="2015-01-01", holdout_topic=None, group_col="ma_id",
    )
    write_validation_report(report, models_dir / "validation_report_temporal.json")
    print(f"      raw AUC {report.auc:.3f}  Brier {report.brier:.3f}  "
          f"slope {report.calibration_slope:.2f}  n_test {report.n_test}")

    print("[4/6] Calibration: isotonic primary, Platt sigmoid sensitivity")
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import clone
    train, _ = split_temporal(
        feats, cutoff="2015-01-01", holdout_topic=None, group_col="ma_id",
    )
    X_tr = train[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
    y_tr = train["flip"].astype(int).values
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

    def _fit_and_validate(method: str) -> tuple[CalibratedClassifierCV, object]:
        cal = CalibratedClassifierCV(base_pipeline, method=method, cv=5)
        cal.fit(X_tr, y_tr)
        bundle = {"pipeline": cal, "features": FEATURE_COLS,
                  "schema_version": "1.0.0", "calibration_method": method}
        rep = validate_model(
            bundle, feats, cutoff="2015-01-01", holdout_topic=None, group_col="ma_id",
        )
        return cal, bundle, rep

    iso_cal, iso_bundle, iso_report = _fit_and_validate("isotonic")
    sig_cal, sig_bundle, sig_report = _fit_and_validate("sigmoid")

    write_validation_report(
        iso_report, models_dir / "validation_report_temporal_isotonic.json",
    )
    write_validation_report(
        sig_report, models_dir / "validation_report_temporal_platt.json",
    )

    print(f"      isotonic   AUC {iso_report.auc:.3f}  Brier {iso_report.brier:.3f}  "
          f"slope {iso_report.calibration_slope:.2f}  intercept {iso_report.calibration_intercept:.2f}")
    print(f"      Platt      AUC {sig_report.auc:.3f}  Brier {sig_report.brier:.3f}  "
          f"slope {sig_report.calibration_slope:.2f}  intercept {sig_report.calibration_intercept:.2f}")

    # Pre-registered slope window: ship the calibrator inside it; if neither
    # is in-window, ship the one closer to slope=1.0 and declare deviation
    # (review-findings option I+III).
    SLOPE_LO, SLOPE_HI = 0.8, 1.2
    iso_passes = SLOPE_LO <= iso_report.calibration_slope <= SLOPE_HI
    sig_passes = SLOPE_LO <= sig_report.calibration_slope <= SLOPE_HI
    if iso_passes and not sig_passes:
        cal_bundle, cal_report, ship_method = iso_bundle, iso_report, "isotonic"
        deviation_note = ""
    elif sig_passes and not iso_passes:
        cal_bundle, cal_report, ship_method = sig_bundle, sig_report, "sigmoid"
        deviation_note = ""
    elif iso_passes and sig_passes:
        # Both pass — prefer isotonic (more flexible).
        cal_bundle, cal_report, ship_method = iso_bundle, iso_report, "isotonic"
        deviation_note = ""
    else:
        # Neither passes — pick the slope closer to 1.0.
        iso_dist = abs(iso_report.calibration_slope - 1.0)
        sig_dist = abs(sig_report.calibration_slope - 1.0)
        if sig_dist < iso_dist:
            cal_bundle, cal_report, ship_method = sig_bundle, sig_report, "sigmoid"
        else:
            cal_bundle, cal_report, ship_method = iso_bundle, iso_report, "isotonic"
        deviation_note = (
            f"DEVIATION: pre-registered slope window [{SLOPE_LO}, {SLOPE_HI}] "
            f"missed by both calibrators (isotonic {iso_report.calibration_slope:.2f}, "
            f"Platt {sig_report.calibration_slope:.2f}). Shipping {ship_method} "
            f"as closest-to-1.0; see manuscript Methods for declared deviation."
        )
        print(f"      {deviation_note}")

    print(f"      shipped: {ship_method}  AUC {cal_report.auc:.3f}  "
          f"slope {cal_report.calibration_slope:.2f}")
    with open(artifacts.gbm_path, "wb") as f:
        pickle.dump(cal_bundle, f)
    write_validation_report(
        cal_report, models_dir / "validation_report_temporal_calibrated.json",
    )

    print("[5/6] Permutation sanity check (5 shuffles on calibrated model)")
    from sklearn.metrics import roc_auc_score
    _, test = split_temporal(
        feats, cutoff="2015-01-01", holdout_topic=None, group_col="ma_id",
    )
    X = test[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
    y = test["flip"].values
    p = cal_bundle["pipeline"].predict_proba(X)[:, 1]
    rng = np.random.default_rng(0)
    perm_aucs = [roc_auc_score(rng.permutation(y), p) for _ in range(5)]
    print(f"      permutation AUC mean {np.mean(perm_aucs):.3f} "
          f"(should be near 0.50)")

    print("[6/6] Ablation: pipeline-features family delta-AUC with paired bootstrap CI")
    from sklearn.metrics import brier_score_loss

    def _ablate_predict(drop_cols: set) -> np.ndarray:
        """Refit calibrated model with `drop_cols` removed; return test probs."""
        keep = [c for c in FEATURE_COLS if c not in drop_cols]
        X_tr_a = train[keep].apply(pd.to_numeric, errors="coerce").values
        X_te_a = test[keep].apply(pd.to_numeric, errors="coerce").values
        cl = CalibratedClassifierCV(
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()),
                      ("clf", clone(base_clf))]),
            method="sigmoid", cv=5,
        )
        cl.fit(X_tr_a, y_tr)
        return cl.predict_proba(X_te_a)[:, 1]

    def _paired_bootstrap_delta(
        y_te: np.ndarray, p_full: np.ndarray, p_ablated: np.ndarray,
        n_boot: int = 1000, seed: int = 0,
    ) -> tuple[float, float, float]:
        """Return (mean_delta, ci_low, ci_high) for AUC(full) - AUC(ablated)."""
        rs = np.random.default_rng(seed)
        n = len(y_te)
        deltas = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rs.integers(0, n, size=n)
            yb = y_te[idx]
            if len(np.unique(yb)) < 2:
                deltas[b] = np.nan
                continue
            deltas[b] = roc_auc_score(yb, p_full[idx]) - roc_auc_score(yb, p_ablated[idx])
        deltas = deltas[~np.isnan(deltas)]
        return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    pipeline_cols = {"pipeline_trial_count", "pipeline_expected_n",
                     "pipeline_sponsor_entropy", "pipeline_design_het",
                     "pipeline_empty"}
    temporal_cols = {"v1_year_span", "v1_years_since_recent", "v1_annual_accrual"}

    p_full = cal_bundle["pipeline"].predict_proba(X)[:, 1]
    full_auc = roc_auc_score(y, p_full)
    full_brier = brier_score_loss(y, p_full)
    print(f"      {'full (all features)':30s} AUC {full_auc:.3f}  Brier {full_brier:.3f}")

    ablation_results = []
    for drop_cols, label in [
        (pipeline_cols, "no pipeline (5 dropped)"),
        (temporal_cols, "no temporal (3 dropped)"),
        (pipeline_cols | temporal_cols, "no pipe+temp (8 dropped)"),
    ]:
        p_a = _ablate_predict(drop_cols)
        auc_a = roc_auc_score(y, p_a)
        brier_a = brier_score_loss(y, p_a)
        delta_mean, ci_low, ci_high = _paired_bootstrap_delta(y, p_full, p_a)
        crosses_zero = ci_low <= 0 <= ci_high
        verdict = "ns" if crosses_zero else ("benefit" if delta_mean > 0 else "harm")
        print(f"      ablated: {label:24s} AUC {auc_a:.3f}  Brier {brier_a:.3f}  "
              f"deltaAUC {delta_mean:+.4f} [95% CI {ci_low:+.4f}, {ci_high:+.4f}]  {verdict}")
        ablation_results.append({
            "label": label, "auc": auc_a, "brier": brier_a,
            "delta_auc_mean": delta_mean, "delta_auc_ci_low": ci_low,
            "delta_auc_ci_high": ci_high, "crosses_zero": crosses_zero,
        })

    import json as _json
    (models_dir / "ablation_report_temporal.json").write_text(_json.dumps({
        "n_test": int(len(y)),
        "n_bootstrap": 1000,
        "full_auc": full_auc,
        "full_brier": full_brier,
        "ablations": ablation_results,
    }, indent=2))

    print("Done. Calibrated model at", artifacts.gbm_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
