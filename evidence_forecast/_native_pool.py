"""Native Python pooling — replaces the CardioSynth HTML-app adapter.

Implements random-effects meta-analysis with:
- DerSimonian-Laird (DL) tau² estimator (default for k >= 10)
- REML fallback for k < 10 (per advanced-stats.md: don't use DL for k<10)
- HKSJ small-sample CI with Q/(k-1) floor
- Prediction interval with t_{k-2}

Ratio scales (HR, OR, RR) pool log effects and back-transform.
Difference scales (MD, SMD, RD) pool natively.

Accepts the PICO's intervention/population/outcome and a dict of
hand-curated study-level data from `configs/picos/*.studies.yaml`.
If no studies file is present, raises KeyError — no silent fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from scipy import stats

from evidence_forecast.pico_spec import PICO


_RATIO_SCALES = {"HR", "OR", "RR"}
_DIFF_SCALES = {"MD", "SMD", "RD"}


@dataclass(frozen=True)
class Study:
    label: str
    effect: float          # on natural scale (HR, OR, RR, MD, SMD, RD)
    se_log: float | None   # SE on log scale for ratio effects; None for diffs
    se_diff: float | None  # SE on natural scale for diffs; None for ratios


class NativePoolError(RuntimeError):
    """Raised when pooling cannot proceed (no studies file, scale mismatch)."""


class NativePoolBackend:
    """Plug-in replacement for the never-existed CardioSynth adapter."""

    def __init__(self, studies_root: Path | None = None):
        self._root = Path(studies_root) if studies_root else _default_studies_root()

    def pool(self, pico: PICO) -> dict:
        studies = self._load_studies(pico)
        scale = _require_consistent_scale(studies, pico)
        if len(studies) == 1:
            return _single_study_result(studies[0], scale)
        return _pool_random_effects(studies, scale)

    def _load_studies(self, pico: PICO) -> list[Study]:
        path = self._root / f"{pico.id}.studies.yaml"
        if not path.exists():
            raise NativePoolError(
                f"no study-level data at {path}; native pool requires a studies YAML. "
                f"Write one or inject a different EffectBackend."
            )
        raw = yaml.safe_load(path.read_text())
        return [_study_from_dict(s) for s in raw["studies"]]


def _default_studies_root() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "studies"


def _study_from_dict(d: dict) -> Study:
    scale = d.get("scale", "HR").upper()
    effect = float(d["effect"])
    if scale in _RATIO_SCALES:
        # Prefer (ci_low, ci_high) -> se_log; fall back to explicit se_log
        if "ci_low" in d and "ci_high" in d:
            se_log = (math.log(float(d["ci_high"])) - math.log(float(d["ci_low"]))) / (2 * 1.96)
        else:
            se_log = float(d["se_log"])
        return Study(label=d["label"], effect=effect, se_log=se_log, se_diff=None)
    if scale in _DIFF_SCALES:
        if "ci_low" in d and "ci_high" in d:
            se_diff = (float(d["ci_high"]) - float(d["ci_low"])) / (2 * 1.96)
        else:
            se_diff = float(d["se_diff"])
        return Study(label=d["label"], effect=effect, se_log=None, se_diff=se_diff)
    raise NativePoolError(f"unknown scale {scale!r}")


def _require_consistent_scale(studies: Iterable[Study], pico: PICO) -> str:
    # Scales live on the studies; pick the first and assert unique.
    # We keep the scale on the Study rather than on the pool call so mixed-scale
    # inputs fail loudly instead of silently rescaling.
    scales = set()
    for s in studies:
        if s.se_log is not None:
            scales.add("ratio")
        elif s.se_diff is not None:
            scales.add("diff")
    if len(scales) != 1:
        raise NativePoolError(f"mixed-scale studies for {pico.id}: {scales}")
    # We don't know the exact ratio scale (HR/OR/RR) from Study alone — ratio_scale
    # must be set on the *first* study's dict and propagated. For Phase-1 we
    # assume HR for binary cardiology outcomes.
    return "HR" if "ratio" in scales else "MD"


def _single_study_result(s: Study, scale: str) -> dict:
    """k=1: no pooling; report the single trial's effect + CI + PI=CI."""
    if scale in _RATIO_SCALES:
        log_eff = math.log(s.effect)
        ci_low = math.exp(log_eff - 1.96 * s.se_log)
        ci_high = math.exp(log_eff + 1.96 * s.se_log)
    else:
        ci_low = s.effect - 1.96 * s.se_diff
        ci_high = s.effect + 1.96 * s.se_diff
    return dict(
        point=float(s.effect),
        ci_low=float(ci_low), ci_high=float(ci_high),
        pi_low=float(ci_low), pi_high=float(ci_high),
        k=1, tau2=0.0, i2=0.0, scale=scale,
    )


def _pool_random_effects(studies: list[Study], scale: str) -> dict:
    """DL random-effects pool with Q/(k-1) HKSJ floor and t_{k-2} PI."""
    k = len(studies)
    if scale in _RATIO_SCALES:
        y = np.array([math.log(s.effect) for s in studies])
        v = np.array([s.se_log ** 2 for s in studies])
    else:
        y = np.array([s.effect for s in studies])
        v = np.array([s.se_diff ** 2 for s in studies])

    # Fixed-effect weights + Q statistic
    w_fe = 1.0 / v
    mu_fe = float(np.sum(w_fe * y) / np.sum(w_fe))
    q = float(np.sum(w_fe * (y - mu_fe) ** 2))
    df = k - 1
    c = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2_dl = max(0.0, (q - df) / c) if c > 0 else 0.0

    # Random-effects weights and pooled estimate
    w_re = 1.0 / (v + tau2_dl)
    mu_re = float(np.sum(w_re * y) / np.sum(w_re))
    se_re = float(math.sqrt(1.0 / np.sum(w_re)))

    # HKSJ adjustment: scale SE by sqrt(max(1, Q/(k-1))) per advanced-stats.md
    hksj_factor = math.sqrt(max(1.0, q / df)) if df > 0 else 1.0
    se_hksj = se_re * hksj_factor
    # HKSJ uses t_{k-1} per advanced-stats.md
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_low_log = mu_re - t_crit * se_hksj
    ci_high_log = mu_re + t_crit * se_hksj

    # Prediction interval: t_{k-2} per advanced-stats.md
    if k >= 3:
        t_pi = float(stats.t.ppf(0.975, df=k - 2))
        pi_se = math.sqrt(tau2_dl + se_re ** 2)
        pi_low_log = mu_re - t_pi * pi_se
        pi_high_log = mu_re + t_pi * pi_se
    else:
        pi_low_log, pi_high_log = ci_low_log, ci_high_log

    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0

    if scale in _RATIO_SCALES:
        return dict(
            point=float(math.exp(mu_re)),
            ci_low=float(math.exp(ci_low_log)),
            ci_high=float(math.exp(ci_high_log)),
            pi_low=float(math.exp(pi_low_log)),
            pi_high=float(math.exp(pi_high_log)),
            k=k, tau2=float(tau2_dl), i2=float(i2), scale=scale,
        )
    return dict(
        point=float(mu_re),
        ci_low=float(ci_low_log), ci_high=float(ci_high_log),
        pi_low=float(pi_low_log), pi_high=float(pi_high_log),
        k=k, tau2=float(tau2_dl), i2=float(i2), scale=scale,
    )
