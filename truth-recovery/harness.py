"""harness.py — truth-recovery driver for evidence-forecast (calibration test).

Procedure
---------
1. Generate a seeded known-truth dataset (dgp_flip) with a KNOWN logistic flip law.
2. Train the engine's REAL flip-forecaster (calibration.train.train_models) on the
   temporal-train split (group-aware off here — synthetic ma_ids are unique).
3. Validate on the held-out temporal split using the engine's REAL validate_model,
   which yields AUC, Brier, calibration slope/intercept, and 10-bin reliability.
4. CALIBRATION MEASUREMENT: is predicted P(flip) calibrated to the realised flip
   rate? We report the engine's calibration slope (target ~1.0), Brier, and the
   max per-bin |predicted - observed| gap across populated reliability bins.
5. ORACLE BENCHMARK: the Brier of the true generative probability (true_p) is the
   irreducible floor; the model's Brier should be close to it, not far below
   (which would signal over-fit leakage) nor far above (mis-calibration).
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine          # noqa: E402  (re-exports train_models / validate_model)
import dgp_flip        # noqa: E402


def run(n=4000, cutoff="2021-01-01", seed=0):
    df = dgp_flip.make_dataset(n=n, seed=seed)

    # held-out oracle Brier floor (on the test side of the temporal split)
    test_mask = df["v1_date"] >= cutoff
    oracle_brier = brier_score_loss(
        df.loc[test_mask, "flip"].values, df.loc[test_mask, "true_p"].values
    )

    feat_df = df.drop(columns=["true_p"])  # engine must not see the truth column

    with tempfile.TemporaryDirectory() as tmp:
        arts = engine.train_models(
            feat_df, models_dir=Path(tmp),
            cutoff=cutoff, holdout_topic=None, seed=seed, group_col=None,
        )
        import pickle
        with open(arts.gbm_path, "rb") as f:
            bundle = pickle.load(f)
        report = engine.validate_model(
            bundle, feat_df, cutoff=cutoff, holdout_topic=None, seed=seed, group_col=None,
        )

    # max per-bin calibration gap over populated bins with >= 20 obs
    gaps = [
        abs(b["predicted_mean"] - b["observed_mean"])
        for b in report.reliability_bins
        if b["count"] >= 20 and b["predicted_mean"] is not None
    ]
    max_bin_gap = max(gaps) if gaps else float("nan")

    # --- production calibration path (CalibratedClassifierCV), as in
    #     scripts/train_on_temporal_pairs.py: the library train_models() ships the
    #     RAW GBM; post-hoc isotonic/Platt calibration is applied only in the
    #     production script. We reproduce it here to fairly measure the served model.
    import pandas as _pd
    cal_reports = {}
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        train, _ = engine.split_temporal(feat_df, cutoff=cutoff, holdout_topic=None, group_col=None)
        feats_cols = [c for c in feat_df.columns
                      if c not in {"ma_id", "flip", "topic_area", "v1_date"}]
        X_tr = train[feats_cols].apply(_pd.to_numeric, errors="coerce").values
        y_tr = train["flip"].astype(int).values
        base = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=seed)),
        ])
        for method in ("isotonic", "sigmoid"):
            cal = CalibratedClassifierCV(base, method=method, cv=5)
            cal.fit(X_tr, y_tr)
            b = {"pipeline": cal, "features": feats_cols, "schema_version": "1.0.0"}
            rep = engine.validate_model(b, feat_df, cutoff=cutoff,
                                        holdout_topic=None, seed=seed, group_col=None)
            cal_reports[method] = {
                "auc": rep.auc, "brier": rep.brier,
                "slope": rep.calibration_slope, "intercept": rep.calibration_intercept,
            }
    except Exception as e:  # pragma: no cover
        cal_reports["error"] = str(e)

    return {
        "n_test": report.n_test,
        "auc": report.auc,
        "brier": report.brier,
        "oracle_brier": float(oracle_brier),
        "calibration_slope": report.calibration_slope,
        "calibration_intercept": report.calibration_intercept,
        "max_bin_gap": max_bin_gap,
        "reliability_bins": report.reliability_bins,
        "calibrated": cal_reports,
    }


def main():
    r = run()
    print("=== evidence-forecast truth-recovery: flip-probability calibration ===")
    print(f"known-truth logistic flip law; n_test={r['n_test']}")
    print(f"AUC                  = {r['auc']:.4f}  (discrimination; truth is signal-bearing)")
    print(f"Brier (model)        = {r['brier']:.4f}")
    print(f"Brier (oracle floor) = {r['oracle_brier']:.4f}  (true_p; irreducible)")
    print(f"calibration slope    = {r['calibration_slope']:.4f}  (target ~1.0)")
    print(f"calibration intercept= {r['calibration_intercept']:.4f}  (target ~0.0)")
    print(f"max per-bin gap      = {r['max_bin_gap']:.4f}  (|predicted-observed|, bins n>=20)")
    print()
    print("reliability bins (predicted -> observed, n):")
    for b in r["reliability_bins"]:
        if b["count"] >= 20 and b["predicted_mean"] is not None:
            print(f"  [{b['bin_low']:.1f},{b['bin_high']:.1f})  pred={b['predicted_mean']:.3f}  obs={b['observed_mean']:.3f}  n={b['count']}")
    print()
    print("production post-hoc calibration (CalibratedClassifierCV, as in train_on_temporal_pairs.py):")
    for m, v in r.get("calibrated", {}).items():
        if isinstance(v, dict) and "slope" in v:
            print(f"  {m:9s} AUC={v['auc']:.3f}  Brier={v['brier']:.3f}  slope={v['slope']:.3f}  intercept={v['intercept']:.3f}")
        else:
            print(f"  {m}: {v}")


if __name__ == "__main__":
    main()
