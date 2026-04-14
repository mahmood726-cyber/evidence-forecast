"""Validation: AUC, Brier, reliability, permutation test, pipeline ablation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from evidence_forecast.calibration.train import split_temporal, _feature_cols


@dataclass
class ValidationReport:
    auc: float
    brier: float
    calibration_slope: float
    calibration_intercept: float
    reliability_bins: list[dict]
    n_test: int


def validate_model(
    bundle: dict,
    df: pd.DataFrame,
    cutoff: str = "2023-01-01",
    holdout_topic: str | None = "cardiology",
    seed: int = 0,
) -> ValidationReport:
    _, test = split_temporal(df, cutoff=cutoff, holdout_topic=None)
    # Holdout eval is on topic == holdout_topic if requested (cardiology gen eval)
    if holdout_topic:
        test = test[test["topic_area"] == holdout_topic].copy()
    features = bundle["features"]
    X = test[features].apply(pd.to_numeric, errors="coerce").values
    y = test["flip"].values
    proba = bundle["pipeline"].predict_proba(X)[:, 1]

    auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else float("nan")
    brier = brier_score_loss(y, proba)
    slope, intercept = _calibration_line(y, proba)
    bins = _reliability_bins(y, proba, n_bins=10)
    return ValidationReport(auc, brier, slope, intercept, bins, len(y))


def _calibration_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    logit_p = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    lr = LogisticRegression(C=1e9, max_iter=500)
    lr.fit(logit_p.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _reliability_bins(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1]) if i < n_bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if mask.any():
            out.append(dict(
                bin_low=float(edges[i]), bin_high=float(edges[i + 1]),
                predicted_mean=float(p[mask].mean()),
                observed_mean=float(y[mask].mean()),
                count=int(mask.sum()),
            ))
        else:
            out.append(dict(
                bin_low=float(edges[i]), bin_high=float(edges[i + 1]),
                predicted_mean=None, observed_mean=None, count=0,
            ))
    return out


def pipeline_ablation(full_report: ValidationReport, ablated_report: ValidationReport) -> float:
    """Return ΔAUC attributable to pipeline features; spec ships-threshold 0.02."""
    return full_report.auc - ablated_report.auc


def write_validation_report(report: ValidationReport, path: Path) -> None:
    Path(path).write_text(json.dumps(asdict(report), indent=2))
