"""Pipeline layer: extracts ongoing-trial features matching a PICO.

Reads AACT snapshot at a given date. Matching is heuristic string-containment
on intervention AND condition. For production use against full AACT this
would expand to MeSH-backed matching; Phase-1 scope uses string match with
the PICO's intervention/population terms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from evidence_forecast.pico_spec import PICO

_ONGOING_STATUSES = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING"}


@dataclass(frozen=True)
class PipelineFeatures:
    trial_count: int
    expected_n_sum: int
    mean_expected_event_rate: float
    sponsor_entropy: float
    design_heterogeneity: float
    pipeline_empty: bool


def extract_pipeline(
    pico: PICO,
    snapshot_date: str,
    aact_path: Path,
    event_rate_default: float = 0.05,
) -> PipelineFeatures:
    aact_path = Path(aact_path)
    if not aact_path.exists():
        raise FileNotFoundError(f"AACT path not found: {aact_path}")
    df = pd.read_csv(aact_path)
    snap = date.fromisoformat(snapshot_date)

    iv_term = getattr(pico, "match_intervention", None) or _primary_token(pico.intervention)
    cd_term = getattr(pico, "match_condition", None) or _primary_token(pico.population)
    matches = df[
        df["interventions"].fillna("").str.lower().str.contains(iv_term.lower(), na=False, regex=False)
        & df["conditions"].fillna("").str.lower().str.contains(cd_term.lower(), na=False, regex=False)
        & df["overall_status"].isin(_ONGOING_STATUSES)
        & (pd.to_datetime(df["start_date"], errors="coerce").dt.date <= snap)
        & (pd.to_datetime(df["completion_date"], errors="coerce").dt.date > snap)
    ].copy()

    if matches.empty:
        return PipelineFeatures(
            trial_count=0, expected_n_sum=0,
            mean_expected_event_rate=0.0,
            sponsor_entropy=0.0, design_heterogeneity=0.0,
            pipeline_empty=True,
        )

    trial_count = len(matches)
    n_sum = int(matches["enrollment"].fillna(0).sum())
    sponsor_entropy = _shannon_entropy(matches["lead_sponsor"].fillna("UNKNOWN").tolist())
    design_heterogeneity = _design_heterogeneity(matches)

    return PipelineFeatures(
        trial_count=trial_count,
        expected_n_sum=n_sum,
        mean_expected_event_rate=event_rate_default,
        sponsor_entropy=sponsor_entropy,
        design_heterogeneity=design_heterogeneity,
        pipeline_empty=False,
    )


def _primary_token(s: str) -> str:
    """First alphabetic word, lowercased, for naive matching."""
    for tok in s.lower().split():
        cleaned = "".join(c for c in tok if c.isalpha())
        if cleaned:
            return cleaned
    return s.lower()


def _shannon_entropy(items: Iterable[str]) -> float:
    items = list(items)
    n = len(items)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _design_heterogeneity(df: pd.DataFrame) -> float:
    combos = df[["study_type", "phase", "primary_purpose"]].fillna("NA").apply(
        lambda r: "|".join(r.astype(str)), axis=1
    )
    return len(combos.unique()) / len(combos)
