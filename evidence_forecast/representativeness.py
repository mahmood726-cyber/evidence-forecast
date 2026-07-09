"""Representativeness layer: burden-weighted overlap between trial-country
mix and disease-burden mix, per registry-first population contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_TOL = 1e-6


@dataclass(frozen=True)
class RepresentativenessResult:
    overlap_score: float
    trial_country_count: int
    burden_weighted: bool
    source: str  # "registry_first", "aact", "ctgov_fallback", "none"


def compute_representativeness(
    trial_country_weights: dict[str, float],
    burden_weights: dict[str, float],
    source: str = "aact",
) -> RepresentativenessResult:
    if not trial_country_weights:
        return RepresentativenessResult(0.0, 0, False, "none")
    _check_normalised(trial_country_weights, "trial_country_weights")
    _check_normalised(burden_weights, "burden_weights")
    overlap = sum(
        min(trial_country_weights.get(k, 0.0), burden_weights.get(k, 0.0))
        for k in set(trial_country_weights) | set(burden_weights)
    )
    return RepresentativenessResult(
        overlap_score=float(overlap),
        trial_country_count=len(trial_country_weights),
        burden_weighted=True,
        source=source,
    )


def _check_normalised(w: dict[str, float], name: str) -> None:
    total = sum(w.values())
    if not math.isclose(total, 1.0, abs_tol=_TOL):
        raise ValueError(f"{name} must sum to 1.0 (±{_TOL}); got {total:.6f}")
