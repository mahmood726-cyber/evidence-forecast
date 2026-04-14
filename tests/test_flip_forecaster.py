import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from evidence_forecast.flip_forecaster import (
    predict_flip, FlipForecast, FlipForecasterError,
)
from evidence_forecast.calibration.train import train_models


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
