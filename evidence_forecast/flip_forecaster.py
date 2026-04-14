"""Inference loader for the flip-forecaster.

Loads persisted GBM pipeline, validates feature names match training,
returns point estimate + bootstrap 95% CI over the predicted probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np

from evidence_forecast.constants import FORECAST_HORIZON_MONTHS


class FlipForecasterError(RuntimeError):
    """Raised on schema mismatch or missing artifact."""


@dataclass(frozen=True)
class FlipForecast:
    probability: float
    ci_low: float
    ci_high: float
    horizon_months: int
    model_version: str
    pipeline_empty: bool


def predict_flip(
    features: dict, model_path: Path,
    bootstrap_n: int = 1000, seed: int = 0,
) -> FlipForecast:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"flip-forecaster model not found: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    expected = bundle["features"]
    missing = [c for c in expected if c not in features]
    if missing:
        raise FlipForecasterError(
            f"features missing {missing}; received keys: {sorted(features.keys())}"
        )
    X = np.array([[float(features[c]) if not isinstance(features[c], bool) else int(features[c])
                   for c in expected]])
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    ci_low, ci_high = _bootstrap_ci(bundle["pipeline"], X, p, bootstrap_n, seed)
    return FlipForecast(
        probability=p, ci_low=ci_low, ci_high=ci_high,
        horizon_months=FORECAST_HORIZON_MONTHS,
        model_version=bundle.get("schema_version", "1.0.0"),
        pipeline_empty=bool(features.get("pipeline_empty", False)),
    )


def _bootstrap_ci(pipeline, X: np.ndarray, p: float, n: int, seed: int) -> tuple[float, float]:
    """For a single observation we approximate the CI by noise-perturbation of inputs.

    NOTE: This is a practical proxy; the calibration-set bootstrap CI is stronger
    but requires the held-out scores. For Phase-1 we use input-perturbation with
    sigma=0.02 on scaled features to convey uncertainty; documented in paper.
    """
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(n):
        noise = rng.normal(0, 0.02, X.shape)
        probs.append(pipeline.predict_proba(X + noise)[0, 1])
    lo, hi = np.quantile(probs, [0.025, 0.975])
    return float(lo), float(hi)
