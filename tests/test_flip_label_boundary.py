"""Flip-label boundary cases and symmetry checks.

Regression guards around label_flips._compute_flip and _null_value — the
"CI crosses null" binary label at the heart of the flip-probability primitive.
Covers edge cases that could cause silent miscounts (CI endpoints exactly at
null, scale casing, flip symmetry).
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast.calibration.label_flips import (
    _compute_flip, _null_value, FlipLabelError,
)


def _row(scale: str, lo1: float, hi1: float, lo2: float, hi2: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "scale": scale,
        "v1_ci_low": lo1, "v1_ci_high": hi1,
        "v2_ci_low": lo2, "v2_ci_high": hi2,
    }])


def test_flip_detected_when_v1_excludes_null_v2_includes():
    # HR: v1 CI [0.6, 0.9] excludes 1.0; v2 [0.7, 1.1] includes 1.0 → flip
    df = _row("HR", 0.6, 0.9, 0.7, 1.1)
    assert _compute_flip(df).iloc[0] == 1


def test_no_flip_when_both_versions_exclude_null_on_same_side():
    # Both CIs entirely below 1.0 — consistent benefit across versions
    df = _row("HR", 0.6, 0.9, 0.65, 0.85)
    assert _compute_flip(df).iloc[0] == 0


def test_no_flip_when_both_versions_include_null():
    df = _row("HR", 0.8, 1.3, 0.7, 1.4)
    assert _compute_flip(df).iloc[0] == 0


def test_flip_detected_from_including_to_excluding():
    # v1 [0.8, 1.2] includes 1.0; v2 [1.05, 1.3] excludes 1.0 → flip
    df = _row("HR", 0.8, 1.2, 1.05, 1.3)
    assert _compute_flip(df).iloc[0] == 1


def test_scale_casing_is_normalised():
    assert _null_value("hr") == 1.0
    assert _null_value("HR") == 1.0
    assert _null_value("Or") == 1.0
    assert _null_value("rr") == 1.0
    assert _null_value("rd") == 0.0
    assert _null_value("SMD") == 0.0


def test_unknown_scale_raises():
    df = _row("XYZ", 0.6, 0.9, 0.7, 1.1)
    with pytest.raises(FlipLabelError):
        _compute_flip(df)


def test_difference_scale_null_is_zero():
    # RD: null=0. v1 [-0.05, 0.02] includes 0; v2 [0.01, 0.08] excludes 0 → flip
    df = _row("RD", -0.05, 0.02, 0.01, 0.08)
    assert _compute_flip(df).iloc[0] == 1


def test_flip_symmetry_under_ci_swap():
    """Swapping v1 ↔ v2 should preserve the flip label (symmetric XOR)."""
    df_a = _row("HR", 0.6, 0.9, 0.7, 1.1)
    df_b = _row("HR", 0.7, 1.1, 0.6, 0.9)
    assert _compute_flip(df_a).iloc[0] == _compute_flip(df_b).iloc[0] == 1


def test_boundary_ci_touching_null_is_not_crossing():
    """CI_high = 1.0 exactly: '<' null fails (not strictly less than), so it
    does NOT cross. This documents the contract — boundary touches count as
    'excludes null'."""
    # v1 [0.6, 1.0] — CI_low=0.6 < 1, CI_high=1.0 not > 1 → not crossing (excludes)
    # v2 [0.8, 1.2] — crosses (includes null)
    df = _row("HR", 0.6, 1.0, 0.8, 1.2)
    # v1 doesn't cross (endpoint at null), v2 crosses → flip
    assert _compute_flip(df).iloc[0] == 1
