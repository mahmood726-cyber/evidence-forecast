"""Label MetaAudit version pairs with binary flip indicator per spec §2.1.

flip = 1 iff sign(CI_low_v1 * CI_high_v1) != sign(CI_low_v2 * CI_high_v2)
i.e., the 95% CI's null-crossing status changes between versions.

Inclusion: gap 6-48mo; both versions report point+CI on same outcome.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

_REQUIRED_COLS = [
    "ma_id", "v1_date", "v2_date", "outcome",
    "v1_point", "v1_ci_low", "v1_ci_high",
    "v2_point", "v2_ci_low", "v2_ci_high",
    "topic_area", "scale",
]

_MIN_GAP_DAYS = 180   # ~6 months
_MAX_GAP_DAYS = 1460  # ~48 months


class FlipLabelError(ValueError):
    """Raised when MetaAudit export fails schema or inclusion checks."""


def label_flips(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise FlipLabelError(
            f"MetaAudit export missing columns {missing}; got {list(df.columns)}"
        )

    df["v1_date"] = pd.to_datetime(df["v1_date"], errors="coerce")
    df["v2_date"] = pd.to_datetime(df["v2_date"], errors="coerce")
    gap = (df["v2_date"] - df["v1_date"]).dt.days

    complete = df[
        df[["v1_point", "v1_ci_low", "v1_ci_high",
            "v2_point", "v2_ci_low", "v2_ci_high"]].notna().all(axis=1)
        & gap.between(_MIN_GAP_DAYS, _MAX_GAP_DAYS)
    ].copy()

    complete["flip"] = _compute_flip(complete)
    return complete.reset_index(drop=True)


def _compute_flip(df: pd.DataFrame) -> pd.Series:
    """CI-crosses-null binary label.

    For ratio scales (HR, OR, RR): null = 1.0, so crossing = (low < 1 < high).
    For difference scales (RD, MD): null = 0.0, so crossing = (low < 0 < high).
    """
    nulls = df["scale"].map(_null_value)
    if nulls.isna().any():
        bad = df[nulls.isna()]["scale"].unique()
        raise FlipLabelError(f"unknown scales: {bad}")
    v1_crosses = (df["v1_ci_low"] < nulls) & (df["v1_ci_high"] > nulls)
    v2_crosses = (df["v2_ci_low"] < nulls) & (df["v2_ci_high"] > nulls)
    return (v1_crosses != v2_crosses).astype(int)


def _null_value(scale: str) -> float | None:
    s = str(scale).upper()
    if s in {"HR", "OR", "RR"}:
        return 1.0
    if s in {"RD", "MD", "SMD"}:
        return 0.0
    return None
