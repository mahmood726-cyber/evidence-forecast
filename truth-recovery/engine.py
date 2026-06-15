"""engine.py — truth-recovery shim for evidence-forecast.

The forecaster's probabilistic core lives in the installed package; we do NOT
re-extract it (that would risk drift). Instead we re-export the exact pure
functions under test so the harness/tests have a single import surface:

  - split_temporal, train_models   (calibration/train.py)  -> the model that emits P(flip)
  - validate_model                 (calibration/validate.py) -> AUC/Brier/reliability/slope
  - predict_flip                   (flip_forecaster.py)      -> the served probability

These are imported VERBATIM from the package; this file only aggregates them.
"""
from __future__ import annotations
import sys
from pathlib import Path

# ensure repo root importable when run from anywhere
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evidence_forecast.calibration.train import (  # noqa: E402
    train_models, split_temporal, FEATURE_COLS_V1,
)
from evidence_forecast.calibration.validate import validate_model  # noqa: E402
from evidence_forecast.flip_forecaster import predict_flip, FlipForecast  # noqa: E402

__all__ = [
    "train_models", "split_temporal", "FEATURE_COLS_V1",
    "validate_model", "predict_flip", "FlipForecast",
]
