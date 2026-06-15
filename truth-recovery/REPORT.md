# Truth-Recovery Report — evidence-forecast

**Repo:** mahmood726-cyber/evidence-forecast
**Engine:** Python package evidence_forecast/ (flip-forecaster: GBM + post-hoc
calibration emitting P(meta-analysis conclusion flips within 24 months))
**Date:** 2026-06-15
**Method:** Re-exported the package's REAL probabilistic functions (train_models,
validate_model, predict_flip) via engine.py; built a seeded known-truth DGP
(dgp_flip.py) where the flip outcome follows a KNOWN logistic law of the v1
features; measured whether predicted P(flip) is CALIBRATED to the realised flip
rate (reliability slope, Brier vs oracle floor, per-bin gap).

## Verdict: GENUINE PROBABILISTIC FORECASTER — CALIBRATION TESTABLE & SUBSTANTIATED (ship)

This is a real probabilistic model, not a deterministic heuristic (contrast the
sibling evidencehalflife, which was flagged for having no probabilistic claim).
It emits a numeric P(flip) in [0,1] with a CI and a 24-month horizon, and the
package itself computes calibration metrics (reliability bins, slope/intercept,
Brier). The calibration claim is therefore directly testable, and it holds when
the production calibration path is used.

## Measured results (seeded known-truth logistic flip law; n_test = 1,812)

| Quantity | Value | Interpretation |
|---|---|---|
| AUC | 0.7749 | recovers the known signal (matches repo's claimed 0.772) |
| Brier (model, raw GBM) | 0.1716 | near oracle floor |
| Brier (oracle floor, true_p) | 0.1525 | irreducible minimum |
| calibration slope (raw GBM) | 0.6805 | UNDER-confident (library default ships raw GBM) |
| max per-bin gap (raw GBM) | 0.1741 | over-predicts at the extremes |

**Production post-hoc calibration (CalibratedClassifierCV, reproduced from
scripts/train_on_temporal_pairs.py):**

| Method | AUC | Brier | slope | intercept |
|---|---|---|---|---|
| isotonic | 0.785 | 0.167 | 0.776 | -0.100 |
| Platt (sigmoid) | 0.786 | 0.165 | **1.060** | 0.114 |

Platt calibration restores calibration (slope 1.06, Brier 0.165 ~ oracle 0.153).
This independently reproduces the repo README's own finding that isotonic and
Platt miss in opposite directions and that Platt lands closer to slope 1.0.

## Honest architectural finding

The post-hoc calibration that makes the forecast trustworthy lives ONLY in the
production script `scripts/train_on_temporal_pairs.py`. The LIBRARY function
`calibration/train.py::train_models()` ships the RAW, UNCALIBRATED GBM (slope
0.68 on the known truth), and `flip_forecaster.py::predict_flip()` serves
whatever bundle it is handed without checking that it was calibrated. Any caller
who trains via the library API and serves that bundle gets an under-calibrated
probability. The released `outputs/` cards use the production-script Platt model,
so they are fine; the risk is library mis-use, not the shipped cards.

A secondary note: `predict_flip._bootstrap_ci` derives its CI by input-noise
perturbation (sigma=0.02), not a calibration-set bootstrap. It is documented as a
proxy in-code; it conveys uncertainty but is not a true sampling CI.

## Recommendation

- SHIP this validation. The probabilistic claim is genuine and calibration is
  substantiated under the production calibration path.
- **Library hardening (non-blocking):** move the CalibratedClassifierCV wrapper
  out of the script and into `train_models()` (or have `predict_flip` refuse a
  bundle lacking a `calibration_method` key), so the served probability is
  calibrated by construction regardless of caller path.

## Files
- engine.py — re-exports real package functions (train_models, validate_model, predict_flip)
- dgp_flip.py — seeded known-truth logistic flip DGP
- harness.py — trains real model on known truth, measures calibration + runs production Platt/isotonic path
- test_truth_recovery.py — 5 assertions (all PASS)
- REPORT.md — this file
