"""PICO specification loader and validator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from evidence_forecast.constants import PICO_REQUIRED_FIELDS

_VALID_OUTCOME_TYPES = {"binary", "continuous", "time_to_event"}


class PICOValidationError(ValueError):
    """Raised when a PICO YAML fails validation."""


@dataclass(frozen=True)
class PICO:
    id: str
    title: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    outcome_type: str
    decision_threshold: float
    match_intervention: str | None = None  # optional AACT match term
    match_condition: str | None = None     # optional AACT match term


def load_pico(path: Path) -> PICO:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise PICOValidationError(f"PICO YAML must be a mapping, got {type(data).__name__}")
    missing = [f for f in PICO_REQUIRED_FIELDS if f not in data]
    if missing:
        raise PICOValidationError(f"missing required fields: {missing}")
    if data["outcome_type"] not in _VALID_OUTCOME_TYPES:
        raise PICOValidationError(
            f"outcome_type must be one of {_VALID_OUTCOME_TYPES}, got {data['outcome_type']!r}"
        )
    return PICO(
        id=str(data["id"]),
        title=str(data["title"]),
        population=str(data["population"]),
        intervention=str(data["intervention"]),
        comparator=str(data["comparator"]),
        outcome=str(data["outcome"]),
        outcome_type=str(data["outcome_type"]),
        decision_threshold=float(data["decision_threshold"]),
        match_intervention=str(data["match_intervention"]) if data.get("match_intervention") else None,
        match_condition=str(data["match_condition"]) if data.get("match_condition") else None,
    )
