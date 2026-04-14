"""Effect layer: produces pooled effect + CI + prediction interval.

Default backend calls CardioSynth's pooling entrypoint. Tests inject a
stub backend to avoid depending on CardioSynth install.

Fail-closed: if the backend returns a dict missing any EFFECT_FIELDS
key, raise KeyError with expected-vs-received diff. No silent defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evidence_forecast.constants import EFFECT_FIELDS
from evidence_forecast.pico_spec import PICO


@dataclass(frozen=True)
class EffectResult:
    point: float
    ci_low: float
    ci_high: float
    pi_low: float
    pi_high: float
    k: int
    tau2: float
    i2: float
    scale: str


class EffectBackend(Protocol):
    def pool(self, pico: PICO) -> dict: ...


def compute_effect(pico: PICO, backend: EffectBackend | None = None) -> EffectResult:
    if backend is None:
        backend = _default_backend()
    raw = backend.pool(pico)
    missing = [k for k in EFFECT_FIELDS if k not in raw]
    if missing:
        raise KeyError(
            f"backend returned dict missing {missing}; "
            f"received keys: {sorted(raw.keys())}"
        )
    return EffectResult(
        point=float(raw["point"]),
        ci_low=float(raw["ci_low"]),
        ci_high=float(raw["ci_high"]),
        pi_low=float(raw["pi_low"]),
        pi_high=float(raw["pi_high"]),
        k=int(raw["k"]),
        tau2=float(raw["tau2"]),
        i2=float(raw["i2"]),
        scale=str(raw["scale"]),
    )


def _default_backend() -> EffectBackend:
    # Native Python pool replaces the never-existent CardioSynth Python API.
    # See REAL_PHASE1_BLOCKERS.md for why the original plan's adapter was abandoned.
    from evidence_forecast._native_pool import NativePoolBackend
    return NativePoolBackend()
