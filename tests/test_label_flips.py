from pathlib import Path
import pandas as pd
import pytest
from evidence_forecast.calibration.label_flips import label_flips, FlipLabelError

FIXTURE = Path(__file__).parent / "fixtures" / "metaaudit_pairs_mini.csv"


def test_filter_excludes_out_of_window_pairs():
    labelled = label_flips(FIXTURE)
    # Inclusion: 6 <= gap <= 48mo; both v1/v2 effects present
    ids = set(labelled["ma_id"])
    assert "MA006" not in ids  # too short
    assert "MA007" not in ids  # too long
    assert "MA008" not in ids  # missing v1_point
    assert ids == {"MA001", "MA002", "MA003", "MA004", "MA005"}


def test_flip_label_correct_for_each_included_pair():
    labelled = label_flips(FIXTURE).set_index("ma_id")
    assert labelled.loc["MA001", "flip"] == 1
    assert labelled.loc["MA002", "flip"] == 0
    assert labelled.loc["MA003", "flip"] == 1
    assert labelled.loc["MA004", "flip"] == 0
    assert labelled.loc["MA005", "flip"] == 1


def test_cardiology_tagging_preserved():
    labelled = label_flips(FIXTURE)
    cardio = labelled[labelled["topic_area"] == "cardiology"]
    assert set(cardio["ma_id"]) == {"MA001", "MA004"}


def test_missing_required_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("ma_id,v1_date\nMA001,2020-01-01\n")
    with pytest.raises(FlipLabelError):
        label_flips(bad)
