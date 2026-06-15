"""test_truth_recovery.py — assertions for evidence-forecast calibration.

Run: python truth-recovery/test_truth_recovery.py   (exits 1 on any failure)
  or: pytest truth-recovery/test_truth_recovery.py

Claim under test: the flip-forecaster emits a PROBABILITY of conclusion flip.
Under a seeded known-truth logistic flip law we verify (a) it recovers the
signal (AUC), (b) its Brier is near the irreducible oracle floor, and (c) the
PRODUCTION post-hoc calibration (Platt/sigmoid CalibratedClassifierCV) yields a
calibration slope near 1.0 — i.e. predicted P(flip) is calibrated to the
realised flip rate. The raw library GBM is documented as under-calibrated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

_R = harness.run(n=4000, cutoff="2021-01-01", seed=0)


def test_recovers_signal_auc():
    # Known signal is present; AUC must be well above chance.
    assert _R["auc"] > 0.70, f"AUC {_R['auc']:.3f} below 0.70 (signal not recovered)"


def test_brier_near_oracle_floor():
    # Model Brier should not exceed the irreducible oracle Brier by a wide margin.
    assert _R["brier"] <= _R["oracle_brier"] + 0.05, (
        f"Brier {_R['brier']:.3f} far above oracle floor {_R['oracle_brier']:.3f}"
    )


def test_raw_gbm_is_underconfident():
    # Documented honest finding: the library's raw GBM (train_models) ships
    # under-calibrated (slope < 1). This pins the known defect.
    assert _R["calibration_slope"] < 0.9, (
        f"raw GBM slope {_R['calibration_slope']:.3f} unexpectedly >= 0.9"
    )


def test_production_platt_calibration_recovers():
    # The production Platt/sigmoid path must restore calibration (slope ~1).
    cal = _R["calibrated"].get("sigmoid")
    assert cal is not None, "production sigmoid calibration did not run"
    assert 0.8 <= cal["slope"] <= 1.25, (
        f"Platt-calibrated slope {cal['slope']:.3f} outside [0.8,1.25]"
    )


def test_calibration_improves_over_raw():
    # Platt calibration must move the slope materially closer to 1.0 than raw GBM.
    raw_gap = abs(_R["calibration_slope"] - 1.0)
    platt_gap = abs(_R["calibrated"]["sigmoid"]["slope"] - 1.0)
    assert platt_gap < raw_gap, (
        f"Platt gap {platt_gap:.3f} not better than raw gap {raw_gap:.3f}"
    )


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                fails += 1
    print("\nALL TESTS PASSED" if fails == 0 else f"\n{fails} TEST(S) FAILED")
    sys.exit(1 if fails else 0)
