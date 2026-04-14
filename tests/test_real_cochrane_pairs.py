"""Contract test: the real Cochrane pairs fixture passes flip-label extraction.

Real pairs come from scripts/extract_cochrane_pairs.R over Pairwise70 .rda
files. n=6 at 2026-04-14 (4 multi-pub reviews × unique-update analyses).
This is a sanity anchor, not a training set — all current pairs are flip=0.
"""
from pathlib import Path
import pandas as pd
import pytest
from evidence_forecast.calibration.label_flips import label_flips


FIXTURE = Path(__file__).parent / "fixtures" / "real_cochrane_pairs_v0.csv"


def test_real_pairs_pass_label_extraction():
    df = label_flips(FIXTURE)
    # Seed has 6 rows; extractor will only grow this over time. Guard lower bound.
    assert len(df) >= 6, f"expected >=6 real pairs; got {len(df)}"


def test_real_pairs_have_valid_ci_geometry():
    df = label_flips(FIXTURE)
    # CI_low < point < CI_high for both versions
    assert (df["v1_ci_low"] < df["v1_point"]).all()
    assert (df["v1_point"] < df["v1_ci_high"]).all()
    assert (df["v2_ci_low"] < df["v2_point"]).all()
    assert (df["v2_point"] < df["v2_ci_high"]).all()


def test_real_pairs_preserve_schema():
    df = pd.read_csv(FIXTURE)
    required = {"ma_id", "v1_date", "v2_date", "outcome",
                "v1_point", "v1_ci_low", "v1_ci_high",
                "v2_point", "v2_ci_low", "v2_ci_high",
                "topic_area", "scale"}
    assert required.issubset(set(df.columns))
