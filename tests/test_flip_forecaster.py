import pickle
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest
from evidence_forecast.flip_forecaster import (
    predict_flip, FlipForecast, FlipForecasterError, ModelVersionError,
)
from evidence_forecast.calibration.train import train_models
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def trained_bundle(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "ma_id": [f"M{i:04d}" for i in range(n)],
        "flip": rng.binomial(1, 0.3, n),
        "topic_area": rng.choice(["oncology", "neurology", "gi"], n),
        "v1_date": pd.to_datetime([f"{y}-06-01" for y in rng.integers(2018, 2022, n)]),
        "ci_width": rng.uniform(0.05, 0.8, n),
        "pi_width": rng.uniform(0.1, 1.5, n),
        "distance_from_null": rng.uniform(0, 0.5, n),
        "k": rng.integers(3, 30, n),
        "tau2": rng.uniform(0, 0.15, n),
        "i2": rng.uniform(0, 0.9, n),
        "fragility_index": rng.integers(0, 20, n),
        "egger_p": rng.uniform(0, 1, n),
        "trim_fill_delta": rng.uniform(0, 0.1, n),
        "benford_mad": rng.uniform(0.005, 0.03, n),
        "pipeline_trial_count": rng.integers(0, 10, n),
        "pipeline_expected_n": rng.integers(0, 20000, n),
        "pipeline_sponsor_entropy": rng.uniform(0, 3, n),
        "pipeline_design_het": rng.uniform(0, 1, n),
        "pipeline_empty": rng.choice([True, False], n),
    })
    artifacts = train_models(df, models_dir=tmp_path, holdout_topic=None)
    return artifacts.gbm_path


def test_predict_returns_probability_and_ci(trained_bundle):
    features = dict(
        ci_width=0.3, pi_width=0.8, distance_from_null=0.1,
        k=8, tau2=0.04, i2=0.5,
        fragility_index=5, egger_p=0.2, trim_fill_delta=0.02, benford_mad=0.015,
        pipeline_trial_count=3, pipeline_expected_n=10000,
        pipeline_sponsor_entropy=1.2, pipeline_design_het=0.7,
        pipeline_empty=False,
    )
    f = predict_flip(features, model_path=trained_bundle, bootstrap_n=100, seed=0)
    assert isinstance(f, FlipForecast)
    assert 0.0 <= f.probability <= 1.0
    assert f.ci_low <= f.probability <= f.ci_high


def test_missing_feature_raises(trained_bundle):
    with pytest.raises(FlipForecasterError) as exc:
        predict_flip({"ci_width": 0.3}, model_path=trained_bundle)
    assert "missing" in str(exc.value).lower()


def test_missing_model_artifact_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_flip({}, model_path=tmp_path / "nope.pkl")


def test_predict_suppresses_known_missing_feature_imputer_warning(tmp_path):
    features = [
        "ci_width", "pi_width", "distance_from_null",
        "k", "tau2", "i2",
        "fragility_index", "egger_p", "trim_fill_delta", "benford_mad",
        "pipeline_trial_count", "pipeline_expected_n",
        "pipeline_sponsor_entropy", "pipeline_design_het", "pipeline_empty",
        "v1_year_span", "v1_years_since_recent", "v1_annual_accrual",
    ]
    X = np.zeros((10, len(features)))
    X[:, 1] = np.nan  # pi_width absent across fit set -> sklearn warning on predict without suppression
    y = np.r_[[0] * 5, [1] * 5]
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", DummyClassifier(strategy="prior")),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Skipping features without any observed values: \[.*\]",
            category=UserWarning,
            module=r"sklearn\.impute\._base",
        )
        pipe.fit(X, y)
    bundle_path = tmp_path / "warn.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump({"pipeline": pipe, "features": features, "schema_version": "1.0.0"}, f)

    inputs = dict(
        ci_width=0.3, pi_width=0.8, distance_from_null=0.1,
        k=8, tau2=0.04, i2=0.5,
        fragility_index=5, egger_p=0.2, trim_fill_delta=0.02, benford_mad=0.015,
        pipeline_trial_count=3, pipeline_expected_n=10000,
        pipeline_sponsor_entropy=1.2, pipeline_design_het=0.7,
        pipeline_empty=False,
        v1_year_span=10.0, v1_years_since_recent=0.0, v1_annual_accrual=0.833,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecast = predict_flip(inputs, model_path=bundle_path, bootstrap_n=5, seed=0)

    assert isinstance(forecast, FlipForecast)
    assert not any("Skipping features without any observed values" in str(w.message) for w in caught)


def _full_features():
    return dict(
        ci_width=0.3, pi_width=0.8, distance_from_null=0.1,
        k=8, tau2=0.04, i2=0.5,
        fragility_index=5, egger_p=0.2, trim_fill_delta=0.02, benford_mad=0.015,
        pipeline_trial_count=3, pipeline_expected_n=10000,
        pipeline_sponsor_entropy=1.2, pipeline_design_het=0.7,
        pipeline_empty=False,
    )


def test_non_finite_feature_raises(trained_bundle):
    features = _full_features()
    features["ci_width"] = float("nan")
    with pytest.raises(FlipForecasterError) as exc:
        predict_flip(features, model_path=trained_bundle, bootstrap_n=5)
    assert "finite" in str(exc.value).lower()
    assert "ci_width" in str(exc.value)


def test_non_numeric_feature_raises(trained_bundle):
    features = _full_features()
    features["k"] = "not-a-number"
    with pytest.raises(FlipForecasterError) as exc:
        predict_flip(features, model_path=trained_bundle, bootstrap_n=5)
    assert "not numeric" in str(exc.value).lower()


def test_bad_bootstrap_n_raises(trained_bundle):
    with pytest.raises(FlipForecasterError) as exc:
        predict_flip(_full_features(), model_path=trained_bundle, bootstrap_n=0)
    assert "bootstrap_n" in str(exc.value)


def test_malformed_bundle_raises(tmp_path):
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as f:
        pickle.dump({"pipeline": object()}, f)  # missing 'features'
    with pytest.raises(FlipForecasterError) as exc:
        predict_flip(_full_features(), model_path=path)
    assert "malformed" in str(exc.value).lower()


class _VersionDriftPipeline:
    """Stand-in for a fitted sklearn pipeline whose predict fails on version drift."""

    def predict_proba(self, X):
        raise AttributeError("'SimpleImputer' object has no attribute '_fill_dtype'")


def test_version_drift_predict_raises_model_version_error(tmp_path):
    path = tmp_path / "drift.pkl"
    bundle = {
        "pipeline": _VersionDriftPipeline(),
        "features": list(_full_features().keys()),
        "schema_version": "1.0.0",
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    with pytest.raises(ModelVersionError) as exc:
        predict_flip(_full_features(), model_path=path, bootstrap_n=5)
    msg = str(exc.value).lower()
    assert "scikit-learn" in msg or "sklearn" in msg
    assert "retrain" in msg
