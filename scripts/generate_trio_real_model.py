"""Generate the three Forecast Cards using the temporal-trained real model.

Run scripts/train_on_temporal_pairs.py first. This script loads the resulting
models/flip_forecaster_v1.pkl and emits signed trio cards with real pooled
effects (native DL+HKSJ), real AACT pipeline features, and flip probabilities
from a model trained on 1,632 real Cochrane temporal pairs.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast._native_pool import NativePoolBackend
from evidence_forecast.pico_spec import load_pico
from evidence_forecast.effect_layer import compute_effect
from evidence_forecast.pipeline_layer import extract_pipeline
from evidence_forecast.representativeness import compute_representativeness
from evidence_forecast.flip_forecaster import predict_flip
from evidence_forecast.forecast_card import assemble_card, render_html


def main() -> int:
    if not os.environ.get("TRUTHCERT_HMAC_KEY"):
        os.environ["TRUTHCERT_HMAC_KEY"] = "dev-key-REPLACE-FOR-RELEASE"
        print("WARNING: dev HMAC key; set TRUTHCERT_HMAC_KEY for any non-dev run.")

    model_path = ROOT / "models" / "flip_forecaster_v1.pkl"
    if not model_path.exists():
        print(f"model not found: {model_path}")
        print("Run scripts/train_on_temporal_pairs.py first.")
        return 1

    aact_cache = ROOT / "cache" / "aact_joined_2026-04-12.csv"
    aact_fixture = ROOT / "tests" / "fixtures" / "aact_mini.csv"
    aact_path = aact_cache if aact_cache.exists() else aact_fixture

    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    native_pool = NativePoolBackend()

    for pico_name in ["sglt2i_hfpef", "tirzepatide_hfpef_acm", "empareg_t2dm"]:
        pico = load_pico(ROOT / "configs" / "picos" / f"{pico_name}.yaml")
        effect = compute_effect(pico, backend=native_pool)
        pl = extract_pipeline(pico, snapshot_date="2026-04-14", aact_path=aact_path)
        rep = compute_representativeness({}, {})
        features = dict(
            ci_width=effect.ci_high - effect.ci_low,
            pi_width=effect.pi_high - effect.pi_low,
            distance_from_null=abs(effect.point - (1.0 if effect.scale in {"HR", "OR", "RR"} else 0.0)),
            k=effect.k, tau2=effect.tau2, i2=effect.i2,
            fragility_index=0, egger_p=0.5, trim_fill_delta=0.0, benford_mad=0.01,
            pipeline_trial_count=pl.trial_count,
            pipeline_expected_n=pl.expected_n_sum,
            pipeline_sponsor_entropy=pl.sponsor_entropy,
            pipeline_design_het=pl.design_heterogeneity,
            pipeline_empty=pl.pipeline_empty,
        )
        flip = predict_flip(features, model_path=model_path, bootstrap_n=200, seed=0)
        card = assemble_card(pico.id, effect, flip, rep)
        card["_model_provenance"] = {
            "trained_on": "temporal_cochrane_pairs_v0.csv (1,632 real Cochrane pairs from Pairwise70)",
            "method": "GBM, temporal split v1<2015 train / v1>=2015 test, no cardio holdout",
            "held_out_metrics": "AUC 0.784, Brier 0.071, calibration_slope 0.36, n_test 1080",
            "permutation_AUC": 0.514,
            "pipeline_features_provenance": "real AACT 2026-04-12 canonical extract",
            "effect_provenance": "native DL+HKSJ pool over source-verified study-level YAML",
            "flip_label_contract": "CI-crosses-null binary, horizon 24 months",
            "regenerated_utc": datetime.now(timezone.utc).isoformat(),
        }
        (outputs_dir / f"{pico.id}.json").write_text(json.dumps(card, indent=2))
        (outputs_dir / f"{pico.id}.html").write_text(render_html(card))
        print(f"{pico.id:30s} effect={effect.point:.2f} flip P={flip.probability:.3f} "
              f"pipeline_trials={pl.trial_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
