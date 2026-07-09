import math
from pathlib import Path

from types import SimpleNamespace

from scripts.run_forecast import _build_forecaster_features, _load_weights


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_weights_reads_registry_first_population_run(monkeypatch, tmp_path):
    run_dir = tmp_path / "outputs" / "population_runs" / "hfpef_sglt2"
    _write(
        run_dir / "trial_country_weights.csv",
        "\n".join(
            [
                "trial_id,iso3c,country_name,country_weight,enrollment_weight,source",
                "NCT1,USA,United States,0.5,100,aact",
                "NCT1,GBR,United Kingdom,0.5,50,aact",
                "NCT2,USA,United States,1.0,25,aact",
            ]
        ),
    )
    _write(
        run_dir / "country_rankings.csv",
        "\n".join(
            [
                "iso3c,burden_number,burden_value",
                "USA,300,30",
                "GBR,100,10",
            ]
        ),
    )
    monkeypatch.setenv("REGISTRY_FIRST_ROOT", str(tmp_path))

    trial_country_weights, burden_weights = _load_weights("sglt2i_hfpef")

    assert set(trial_country_weights) == {"USA", "GBR"}
    assert math.isclose(sum(trial_country_weights.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(trial_country_weights["USA"], 125 / 175, abs_tol=1e-9)
    assert math.isclose(trial_country_weights["GBR"], 50 / 175, abs_tol=1e-9)
    assert math.isclose(sum(burden_weights.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(burden_weights["USA"], 0.75, abs_tol=1e-9)
    assert math.isclose(burden_weights["GBR"], 0.25, abs_tol=1e-9)


def test_load_weights_returns_empty_for_unmapped_pico(monkeypatch, tmp_path):
    monkeypatch.setenv("REGISTRY_FIRST_ROOT", str(tmp_path))
    assert _load_weights("empareg_t2dm") == ({}, {})


def test_build_forecaster_features_includes_temporal_defaults():
    effect = SimpleNamespace(
        point=0.81,
        ci_low=0.73,
        ci_high=0.90,
        pi_low=0.61,
        pi_high=1.05,
        k=2,
        tau2=0.0,
        i2=0.0,
        scale="HR",
    )
    pipeline = SimpleNamespace(
        trial_count=2,
        expected_n_sum=604,
        sponsor_entropy=1.0,
        design_heterogeneity=1.0,
        pipeline_empty=False,
    )

    features = _build_forecaster_features(effect=effect, pl=pipeline)

    assert features["v1_year_span"] == 10.0
    assert features["v1_years_since_recent"] == 0.0
    assert math.isclose(features["v1_annual_accrual"], 0.833, abs_tol=1e-12)
