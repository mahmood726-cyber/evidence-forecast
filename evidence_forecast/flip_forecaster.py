"""Inference loader for the flip-forecaster.

Loads persisted GBM pipeline, validates feature names match training,
returns point estimate + bootstrap 95% CI over the predicted probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
import numpy as np

from evidence_forecast.constants import FORECAST_HORIZON_MONTHS


class FlipForecasterError(RuntimeError):
    """Raised on schema mismatch or missing artifact."""


class ModelVersionError(FlipForecasterError):
    """Raised when the persisted model cannot run under the installed sklearn.

    The shipped ``models/flip_forecaster_v1.pkl`` is pickled against a specific
    scikit-learn version. A newer/older installed sklearn can unpickle the
    estimators but then fail deep inside ``predict_proba`` with an opaque
    ``AttributeError`` (e.g. ``'SimpleImputer' object has no attribute
    '_fill_dtype'``). We convert that into an actionable message instead of a
    cryptic stack trace.
    """


@dataclass(frozen=True)
class FlipForecast:
    probability: float
    ci_low: float
    ci_high: float
    horizon_months: int
    model_version: str
    pipeline_empty: bool


_KNOWN_IMPUTER_WARNING = (
    r"Skipping features without any observed values: \[.*\]\. "
    r"At least one non-missing value is needed for imputation with strategy='median'\."
)


def _load_bundle(model_path: Path):
    """Unpickle the model bundle, surfacing a sklearn version mismatch clearly.

    scikit-learn emits an ``InconsistentVersionWarning`` when the estimators
    were pickled under a different version. We capture that warning so a
    downstream predict failure can be attributed to it with an actionable hint.
    """
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:  # pragma: no cover - sklearn always present per deps
        InconsistentVersionWarning = None  # type: ignore[assignment]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(model_path, "rb") as f:
            try:
                bundle = pickle.load(f)
            except (ModuleNotFoundError, AttributeError) as exc:
                raise ModelVersionError(
                    f"cannot unpickle {model_path} with the installed scikit-learn; "
                    "the model was trained under a different version. Retrain via "
                    "`python scripts/train_on_temporal_pairs.py` or install the "
                    f"pinned sklearn. Underlying error: {type(exc).__name__}: {exc}"
                ) from exc
    if InconsistentVersionWarning is not None:
        drift = [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]
        if drift and isinstance(bundle, dict):
            bundle.setdefault("_version_drift", str(drift[0].message))
    return bundle


def _coerce_feature_row(features: dict, expected: list[str]) -> np.ndarray:
    """Build the ordered feature matrix, validating each value is a finite number."""
    row: list[float] = []
    for c in expected:
        v = features[c]
        if isinstance(v, bool):
            row.append(int(v))
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise FlipForecasterError(
                f"feature {c!r} is not numeric: got {v!r} ({type(v).__name__})"
            ) from exc
        if not np.isfinite(fv):
            raise FlipForecasterError(
                f"feature {c!r} must be finite, got {v!r}"
            )
        row.append(fv)
    return np.array([row], dtype=float)


def predict_flip(
    features: dict, model_path: Path,
    bootstrap_n: int = 1000, seed: int = 0,
) -> FlipForecast:
    if bootstrap_n < 1:
        raise FlipForecasterError(f"bootstrap_n must be >= 1, got {bootstrap_n}")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"flip-forecaster model not found: {model_path}")
    bundle = _load_bundle(model_path)
    if not isinstance(bundle, dict) or "features" not in bundle or "pipeline" not in bundle:
        raise FlipForecasterError(
            f"malformed model bundle at {model_path}: "
            "expected a dict with 'pipeline' and 'features' keys"
        )
    expected = bundle["features"]
    missing = [c for c in expected if c not in features]
    if missing:
        raise FlipForecasterError(
            f"features missing {missing}; received keys: {sorted(features.keys())}"
        )
    X = _coerce_feature_row(features, expected)
    p = _predict_probability(bundle["pipeline"], X)
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
        probs.append(_predict_probability(pipeline, X + noise))
    lo, hi = np.quantile(probs, [0.025, 0.975])
    return float(lo), float(hi)


def _predict_probability(pipeline, X: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_KNOWN_IMPUTER_WARNING,
            category=UserWarning,
            module=r"sklearn\.impute\._base",
        )
        try:
            return float(pipeline.predict_proba(X)[0, 1])
        except (AttributeError, TypeError) as exc:
            # A private-attribute AttributeError inside a fitted sklearn estimator
            # is the signature of a pickle/runtime version mismatch (e.g.
            # 'SimpleImputer' object has no attribute '_fill_dtype').
            raise ModelVersionError(
                "flip-forecaster prediction failed inside scikit-learn "
                f"({type(exc).__name__}: {exc}). This almost always means the "
                "persisted model was trained under a different scikit-learn "
                "version than the one installed. Retrain the model via "
                "`python scripts/train_on_temporal_pairs.py`, or install the "
                "scikit-learn version the model was pickled with."
            ) from exc
