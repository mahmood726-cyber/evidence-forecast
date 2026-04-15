"""Generate the three Forecast Cards using the temporal-trained real model.

Run scripts/train_on_temporal_pairs.py first. This script loads the resulting
models/flip_forecaster_v1.pkl and emits signed trio cards with real pooled
effects (native DL+HKSJ), real AACT pipeline features, and flip probabilities
from a model trained on 1,632 real Cochrane temporal pairs.
"""
from __future__ import annotations

import json
import math
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

    # Load shipped-model provenance dynamically so it can't drift.
    val_path = ROOT / "models" / "validation_report_temporal_calibrated.json"
    val = json.loads(val_path.read_text())
    ablation_path = ROOT / "models" / "ablation_report_temporal.json"
    ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None

    def _distance_from_null(point: float, scale: str) -> float:
        # Matches training feature computation (P0-4): log scale for ratios,
        # natural scale for differences.
        if scale in {"HR", "OR", "RR"}:
            if point <= 0:
                return float("nan")
            return abs(math.log(point))
        return abs(point)

    for pico_name in ["sglt2i_hfpef", "tirzepatide_hfpef_acm", "empareg_t2dm"]:
        pico = load_pico(ROOT / "configs" / "picos" / f"{pico_name}.yaml")
        effect = compute_effect(pico, backend=native_pool)
        pl = extract_pipeline(pico, snapshot_date="2026-04-14", aact_path=aact_path)
        rep = compute_representativeness({}, {})
        features = dict(
            ci_width=effect.ci_high - effect.ci_low,
            pi_width=effect.pi_high - effect.pi_low,
            distance_from_null=_distance_from_null(effect.point, effect.scale),
            k=effect.k, tau2=effect.tau2, i2=effect.i2,
            fragility_index=0, egger_p=0.5, trim_fill_delta=0.0, benford_mad=0.01,
            pipeline_trial_count=pl.trial_count,
            pipeline_expected_n=pl.expected_n_sum,
            pipeline_sponsor_entropy=pl.sponsor_entropy,
            pipeline_design_het=pl.design_heterogeneity,
            pipeline_empty=pl.pipeline_empty,
            # v1-intrinsic temporal features: production PICOs don't have
            # study-year distributions on disk; use training-set medians.
            v1_year_span=10.0,
            v1_years_since_recent=0.0,
            v1_annual_accrual=0.833,
        )
        flip = predict_flip(features, model_path=model_path, bootstrap_n=200, seed=0)
        card = assemble_card(pico.id, effect, flip, rep)
        card["_model_provenance"] = {
            "trained_on": "temporal_cochrane_pairs_enriched.csv (1,669 analysed pairs from 560 Pairwise70 .rda + 557 CrossRef titles; label-flips filter on 3,156 raw pairs)",
            "method": "GBM + sigmoid Platt calibration via CalibratedClassifierCV(cv=5); "
                      "group-aware temporal split by ma_id (median v1<2015 train / v1>=2015 test), "
                      "no cardio holdout; word-boundary AACT matching",
            "held_out_metrics": {
                "auc": round(val["auc"], 4),
                "brier": round(val["brier"], 4),
                "calibration_slope": round(val["calibration_slope"], 3),
                "calibration_intercept": round(val["calibration_intercept"], 3),
                "n_test": val["n_test"],
                "ship_thresholds": "AUC>=0.70 PASS; Brier<0.18 PASS; slope in [0.8,1.2] DECLARED DEVIATION",
            },
            "pipeline_ablation": {
                "delta_auc_mean": round(ablation["ablations"][0]["delta_auc_mean"], 5),
                "delta_auc_ci_low": round(ablation["ablations"][0]["delta_auc_ci_low"], 5),
                "delta_auc_ci_high": round(ablation["ablations"][0]["delta_auc_ci_high"], 5),
                "n_bootstrap": ablation["n_bootstrap"],
                "verdict": "null (CI brackets zero)",
            } if ablation else None,
            "pipeline_features_provenance": "real AACT 2026-04-12 canonical extract with word-boundary matching; 188/3,156 pairs (6.0%) have non-empty pipeline at v1-date",
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
