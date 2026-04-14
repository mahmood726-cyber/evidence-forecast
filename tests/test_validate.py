import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from evidence_forecast.calibration.train import train_models
from evidence_forecast.calibration.validate import validate_model, ValidationReport


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(0)
    n = 400
    years = rng.integers(2018, 2025, n)
    dates = pd.to_datetime([f"{y}-06-01" for y in years])
    topics = rng.choice(["cardiology", "oncology", "neurology", "gi"], n, p=[0.2, 0.3, 0.3, 0.2])
    pipeline_n = rng.integers(0, 20000, n)
    p = 0.1 + 0.6 * (pipeline_n / max(pipeline_n.max(), 1))
    flip = rng.binomial(1, p)
    return pd.DataFrame({
        "ma_id": [f"M{i:04d}" for i in range(n)],
        "flip": flip,
        "topic_area": topics,
        "v1_date": dates,
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
        "pipeline_expected_n": pipeline_n,
        "pipeline_sponsor_entropy": rng.uniform(0, 3, n),
        "pipeline_design_het": rng.uniform(0, 1, n),
        "pipeline_empty": rng.choice([True, False], n),
    })


def test_validation_report_has_required_metrics(tmp_path, synthetic_features):
    artifacts = train_models(synthetic_features, models_dir=tmp_path)
    with open(artifacts.gbm_path, "rb") as f:
        bundle = pickle.load(f)
    report = validate_model(
        bundle, synthetic_features, cutoff="2023-01-01",
        holdout_topic=None, seed=0,
    )
    assert isinstance(report, ValidationReport)
    assert 0.0 <= report.auc <= 1.0
    assert 0.0 <= report.brier <= 1.0
    assert len(report.reliability_bins) == 10


def test_permutation_collapses_auc_to_chance(tmp_path, synthetic_features):
    from sklearn.metrics import roc_auc_score
    from evidence_forecast.calibration.train import split_temporal
    artifacts = train_models(synthetic_features, models_dir=tmp_path)
    with open(artifacts.gbm_path, "rb") as f:
        bundle = pickle.load(f)
    rng = np.random.default_rng(1)
    train, test = split_temporal(synthetic_features, cutoff="2023-01-01", holdout_topic=None)
    X = test[bundle["features"]].apply(pd.to_numeric, errors="coerce").values
    y = test["flip"].values
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    aucs = [roc_auc_score(rng.permutation(y), proba) for _ in range(5)]
    mean = float(np.mean(aucs))
    assert 0.30 <= mean <= 0.70, f"permutation AUC {mean:.3f} should collapse to chance"
