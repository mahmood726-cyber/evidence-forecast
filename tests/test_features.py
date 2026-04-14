from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from evidence_forecast.calibration.features import build_features

FIX_DIR = Path(__file__).parent / "fixtures"
AACT = FIX_DIR / "aact_mini.csv"


@pytest.fixture(scope="module")
def pairs_with_extras(tmp_path_factory):
    """Augment the mini MetaAudit fixture with fragility/bias/Benford columns."""
    src = pd.read_csv(FIX_DIR / "metaaudit_pairs_mini.csv")
    src["v1_fragility_index"] = [3, 8, 2, 10, 4, 5, 6, 0]
    src["v1_egger_p"] = [0.12, 0.45, 0.03, 0.70, 0.08, 0.22, 0.33, 0.50]
    src["v1_trim_fill_delta"] = [0.02, 0.00, 0.08, 0.01, 0.03, 0.00, 0.01, 0.00]
    src["v1_benford_mad"] = [0.015, 0.008, 0.025, 0.010, 0.018, 0.009, 0.011, 0.007]
    src["v1_k"] = [6, 9, 5, 12, 8, 4, 10, 6]
    src["v1_tau2"] = [0.04, 0.02, 0.08, 0.01, 0.05, 0.03, 0.02, 0.04]
    src["v1_i2"] = [0.55, 0.30, 0.70, 0.20, 0.60, 0.40, 0.35, 0.50]
    src["v1_population_term"] = "heart failure"
    src["v1_intervention_term"] = "tirzepatide"
    d = tmp_path_factory.mktemp("pairs")
    out = d / "pairs.csv"
    src.to_csv(out, index=False)
    return out


def test_features_returned_per_included_pair(pairs_with_extras):
    features = build_features(pairs_with_extras, aact_path=AACT)
    # Inclusion filter yields 5 rows (MA001-005)
    assert len(features) == 5
    assert set(features.columns) >= {
        "ma_id", "flip", "topic_area",
        "ci_width", "pi_width", "distance_from_null",
        "k", "tau2", "i2",
        "fragility_index", "egger_p", "trim_fill_delta",
        "benford_mad",
        "pipeline_trial_count", "pipeline_expected_n",
        "pipeline_sponsor_entropy", "pipeline_design_het",
        "pipeline_empty",
    }


def test_pipeline_features_use_v1_snapshot(pairs_with_extras):
    features = build_features(pairs_with_extras, aact_path=AACT).set_index("ma_id")
    # MA001 v1_date=2018-06-01: no tirzepatide trials in fixture started before then → empty
    assert features.loc["MA001", "pipeline_empty"] == True
    assert features.loc["MA001", "pipeline_trial_count"] == 0


def test_no_v2_leakage_in_feature_names(pairs_with_extras):
    features = build_features(pairs_with_extras, aact_path=AACT)
    for c in features.columns:
        assert "v2_" not in c, f"leaked v2 feature: {c}"
