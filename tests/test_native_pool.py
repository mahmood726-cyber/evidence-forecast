import math
from pathlib import Path

import pytest

from evidence_forecast._native_pool import (
    NativePoolBackend, NativePoolError, Study,
    _single_study_result, _pool_random_effects,
)
from evidence_forecast.pico_spec import PICO


ROOT = Path(__file__).resolve().parents[1]


def _pico(pid: str) -> PICO:
    return PICO(
        id=pid, title="t", population="p", intervention="i",
        comparator="c", outcome="o", outcome_type="binary",
        decision_threshold=1.0,
    )


def test_single_study_roundtrip():
    s = Study(label="t", effect=0.86, se_log=(math.log(0.99) - math.log(0.74)) / (2 * 1.96), se_diff=None)
    r = _single_study_result(s, scale="HR")
    assert math.isclose(r["point"], 0.86, abs_tol=1e-9)
    assert math.isclose(r["ci_low"], 0.74, abs_tol=0.005)
    assert math.isclose(r["ci_high"], 0.99, abs_tol=0.005)
    assert r["k"] == 1
    assert r["tau2"] == 0.0
    assert r["i2"] == 0.0


def test_two_study_pool_hfpef_sglt2i():
    studies = [
        Study("DELIVER", 0.82, (math.log(0.92) - math.log(0.73)) / (2 * 1.96), None),
        Study("EMPEROR-P", 0.79, (math.log(0.90) - math.log(0.69)) / (2 * 1.96), None),
    ]
    r = _pool_random_effects(studies, scale="HR")
    # Published Vaduganathan 2022 pooled HR ≈ 0.80 (95% CI ~0.73-0.87)
    assert 0.78 <= r["point"] <= 0.83
    assert r["ci_low"] < r["point"] < r["ci_high"]
    assert r["k"] == 2
    assert r["tau2"] >= 0.0
    # With two nearly-consistent studies, I² should be low
    assert r["i2"] < 0.3


def test_backend_loads_studies_yaml_and_pools():
    be = NativePoolBackend(studies_root=ROOT / "configs" / "studies")
    r = be.pool(_pico("sglt2i_hfpef"))
    assert r["k"] == 2
    assert r["scale"] == "HR"
    assert 0.78 <= r["point"] <= 0.83


def test_backend_single_trial_pico():
    be = NativePoolBackend(studies_root=ROOT / "configs" / "studies")
    r = be.pool(_pico("empareg_t2dm"))
    assert r["k"] == 1
    assert math.isclose(r["point"], 0.86, abs_tol=1e-9)


def test_backend_missing_studies_yaml_fails_closed(tmp_path):
    be = NativePoolBackend(studies_root=tmp_path)
    with pytest.raises(NativePoolError) as exc:
        be.pool(_pico("does_not_exist"))
    assert "no study-level data" in str(exc.value).lower()
