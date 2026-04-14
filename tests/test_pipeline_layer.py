from pathlib import Path
import math
import pytest
from evidence_forecast.pipeline_layer import (
    extract_pipeline,
    PipelineFeatures,
)
from evidence_forecast.pico_spec import PICO

FIXTURE = Path(__file__).parent / "fixtures" / "aact_mini.csv"


@pytest.fixture
def tirzepatide_pico() -> PICO:
    return PICO(
        id="tirzepatide_hfpef_acm", title="t",
        population="heart failure", intervention="tirzepatide",
        comparator="placebo", outcome="mortality",
        outcome_type="binary", decision_threshold=1.0,
    )


def test_tirzepatide_pipeline_matches_2_active_hf_trials(tirzepatide_pico):
    feats = extract_pipeline(
        tirzepatide_pico,
        snapshot_date="2024-12-31",
        aact_path=FIXTURE,
    )
    # NCT06000001 + NCT06000002 match (HF + tirzepatide + ongoing at 2024-12-31).
    # NCT06000008 has condition=Diabetes so HF filter excludes it.
    assert feats.trial_count == 2
    assert feats.expected_n_sum == 5500
    assert feats.pipeline_empty is False


def test_sglt2i_pipeline_is_empty_at_2024(tirzepatide_pico):
    sglt2i = PICO(
        id="sglt2i_hfpef", title="t",
        population="heart failure", intervention="sglt2",
        comparator="placebo", outcome="mortality",
        outcome_type="binary", decision_threshold=1.0,
    )
    feats = extract_pipeline(sglt2i, snapshot_date="2024-12-31", aact_path=FIXTURE)
    # Only completed SGLT2 trial in fixture; ongoing count at 2024-12-31 is 0
    assert feats.trial_count == 0
    assert feats.pipeline_empty is True


def test_sponsor_entropy_computed(tirzepatide_pico):
    feats = extract_pipeline(
        tirzepatide_pico, snapshot_date="2024-12-31", aact_path=FIXTURE,
    )
    # All 3 tirzepatide trials are Lilly-sponsored; entropy should be 0
    assert math.isclose(feats.sponsor_entropy, 0.0, abs_tol=1e-9)


def test_design_heterogeneity_for_matched_trials(tirzepatide_pico):
    # Matched trials (NCT001, NCT002) are both INTERVENTIONAL/PHASE3/TREATMENT.
    # 1 distinct combo / 2 trials = 0.5
    feats = extract_pipeline(
        tirzepatide_pico, snapshot_date="2024-12-31", aact_path=FIXTURE,
    )
    assert math.isclose(feats.design_heterogeneity, 0.5, abs_tol=1e-9)


def test_missing_aact_path_fails_closed(tirzepatide_pico, tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pipeline(
            tirzepatide_pico, snapshot_date="2024-12-31",
            aact_path=tmp_path / "nope.csv",
        )
