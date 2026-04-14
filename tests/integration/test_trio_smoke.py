"""Trio smoke test — all three Forecast Cards emit without error.

Runs in <120s (per CLAUDE.md verification preflight contract). Uses the
bundled AACT mini fixture and a stub effect/model backend so it runs even
when CardioSynth and the trained model artifact are not available.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from evidence_forecast.pico_spec import load_pico
from evidence_forecast.effect_layer import compute_effect, EffectBackend
from evidence_forecast.pipeline_layer import extract_pipeline
from evidence_forecast.representativeness import compute_representativeness
from evidence_forecast.flip_forecaster import predict_flip
from evidence_forecast.forecast_card import assemble_card, render_html

ROOT = Path(__file__).resolve().parents[2]
AACT_FIX = ROOT / "tests" / "fixtures" / "aact_mini.csv"


class _StubBackend(EffectBackend):
    def pool(self, pico):
        return dict(
            point=0.82, ci_low=0.72, ci_high=0.94,
            pi_low=0.55, pi_high=1.22,
            k=7, tau2=0.02, i2=0.4, scale="HR",
        )


@pytest.fixture(scope="module")
def stub_model(tmp_path_factory):
    features = [
        "ci_width", "pi_width", "distance_from_null",
        "k", "tau2", "i2",
        "fragility_index", "egger_p", "trim_fill_delta", "benford_mad",
        "pipeline_trial_count", "pipeline_expected_n",
        "pipeline_sponsor_entropy", "pipeline_design_het", "pipeline_empty",
    ]
    pipe = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()),
                     ("clf", DummyClassifier(strategy="prior"))])
    pipe.fit(np.zeros((10, len(features))), np.r_[[0]*5, [1]*5])
    d = tmp_path_factory.mktemp("m")
    path = d / "stub.pkl"
    with open(path, "wb") as f:
        pickle.dump({"pipeline": pipe, "features": features, "schema_version": "1.0.0"}, f)
    return path


@pytest.mark.parametrize("pico_name", ["sglt2i_hfpef", "tirzepatide_hfpef_acm", "empareg_t2dm"])
def test_trio_emits_signed_card(pico_name, seed_env, stub_model):
    pico = load_pico(ROOT / "configs" / "picos" / f"{pico_name}.yaml")
    effect = compute_effect(pico, backend=_StubBackend())
    pl = extract_pipeline(pico, snapshot_date="2024-12-31", aact_path=AACT_FIX)
    rep = compute_representativeness({}, {})
    features = dict(
        ci_width=effect.ci_high - effect.ci_low,
        pi_width=effect.pi_high - effect.pi_low,
        distance_from_null=abs(effect.point - 1.0),
        k=effect.k, tau2=effect.tau2, i2=effect.i2,
        fragility_index=0, egger_p=0.5, trim_fill_delta=0.0, benford_mad=0.01,
        pipeline_trial_count=pl.trial_count,
        pipeline_expected_n=pl.expected_n_sum,
        pipeline_sponsor_entropy=pl.sponsor_entropy,
        pipeline_design_het=pl.design_heterogeneity,
        pipeline_empty=pl.pipeline_empty,
    )
    flip = predict_flip(features, model_path=stub_model, bootstrap_n=50, seed=0)
    card = assemble_card(pico.id, effect, flip, rep)
    assert card["pico_id"] == pico_name
    assert "truthcert" in card
    html = render_html(card)
    assert pico_name in html
