from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from evidence_forecast.calibration.train import (
    split_temporal, train_models, TrainingArtifacts,
)


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    years = rng.integers(2018, 2025, n)
    dates = pd.to_datetime([f"{y}-06-01" for y in years])
    topics = rng.choice(["cardiology", "oncology", "neurology", "gi"], n, p=[0.2, 0.3, 0.3, 0.2])
    # base flip ~20% but pipeline features increase probability
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


def test_temporal_split_respects_cutoff(synthetic_features):
    train, test = split_temporal(synthetic_features, cutoff="2023-01-01")
    assert train["v1_date"].max() < pd.Timestamp("2023-01-01")
    assert test["v1_date"].min() >= pd.Timestamp("2023-01-01")


def test_group_split_no_ma_id_in_both(synthetic_features):
    # P0-5 contract: when group_col='ma_id' is set, no ma_id may straddle.
    df = synthetic_features.copy()
    # Inject 5 reviews with multiple pairs straddling the cutoff.
    extra_rows = []
    for i in range(5):
        ma = f"GROUP{i:03d}"
        for date_str in ["2022-06-01", "2023-06-01", "2024-06-01"]:
            r = df.iloc[0].to_dict()
            r["ma_id"] = ma
            r["v1_date"] = pd.Timestamp(date_str)
            extra_rows.append(r)
    df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    train, test = split_temporal(df, cutoff="2023-01-01", group_col="ma_id")
    overlap = set(train["ma_id"]) & set(test["ma_id"])
    assert overlap == set(), f"ma_id leaked across split: {overlap}"


def test_cardiology_held_out_of_training(synthetic_features):
    train, test = split_temporal(
        synthetic_features, cutoff="2023-01-01", holdout_topic="cardiology"
    )
    assert (train["topic_area"] != "cardiology").all()
    assert "cardiology" in set(test["topic_area"])


def test_train_models_produces_three_fitted_artifacts(tmp_path, synthetic_features):
    artifacts = train_models(
        synthetic_features, models_dir=tmp_path, cutoff="2023-01-01",
        holdout_topic="cardiology",
    )
    assert isinstance(artifacts, TrainingArtifacts)
    assert (tmp_path / "flip_forecaster_v1.pkl").exists()       # GBM shipped
    assert (tmp_path / "flip_forecaster_v1_rf.pkl").exists()
    assert (tmp_path / "flip_forecaster_v1_l1.pkl").exists()
    assert artifacts.feature_names[0] != "ma_id"
    assert "flip" not in artifacts.feature_names
