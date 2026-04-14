"""Dev-mode bootstrap: train a synthetic flip-forecaster and emit trio cards.

Purpose: produce tangible Phase-1 artifacts (signed JSON + HTML cards,
validation report) without CardioSynth and MetaAudit wiring.

Outputs are marked DEV MODE — not a release. Real Task 17 replaces the
synthetic training with a MetaAudit-trained model and CardioSynth effect.

Usage:
    TRUTHCERT_HMAC_KEY=dev-key-only python scripts/dev_bootstrap.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast.calibration.train import train_models
from evidence_forecast.calibration.validate import validate_model, write_validation_report
from evidence_forecast.pico_spec import load_pico
from evidence_forecast.effect_layer import compute_effect
from evidence_forecast._native_pool import NativePoolBackend
from evidence_forecast.pipeline_layer import extract_pipeline
from evidence_forecast.representativeness import compute_representativeness
from evidence_forecast.flip_forecaster import predict_flip
from evidence_forecast.forecast_card import assemble_card, render_html


def _synthetic_training_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = rng.integers(2018, 2025, n)
    dates = pd.to_datetime([f"{y}-06-01" for y in years])
    topics = rng.choice(
        ["cardiology", "oncology", "neurology", "gi", "respiratory", "endocrine"],
        n, p=[0.15, 0.25, 0.2, 0.15, 0.15, 0.1],
    )
    pipeline_n = rng.integers(0, 20000, n)
    # Flip probability rises with pipeline N and with CI-width proximity to null
    ci_width = rng.uniform(0.05, 0.8, n)
    dist = rng.uniform(0, 0.5, n)
    p = 0.1 + 0.4 * (pipeline_n / 20000) + 0.2 * (1 - dist / 0.5) + 0.1 * (1 - ci_width / 0.8)
    p = np.clip(p, 0.0, 0.95)
    flip = rng.binomial(1, p)
    return pd.DataFrame({
        "ma_id": [f"M{i:04d}" for i in range(n)],
        "flip": flip,
        "topic_area": topics,
        "v1_date": dates,
        "ci_width": ci_width,
        "pi_width": rng.uniform(0.1, 1.5, n),
        "distance_from_null": dist,
        "k": rng.integers(3, 30, n),
        "tau2": rng.uniform(0, 0.15, n),
        "i2": rng.uniform(0, 0.9, n),
        "fragility_index": rng.integers(0, 20, n),
        "egger_p": rng.uniform(0, 1, n),
        "trim_fill_delta": rng.uniform(0, 0.1, n),
        "benford_mad": rng.uniform(0.005, 0.03, n),
        "pipeline_trial_count": rng.integers(0, 10, n),
        "pipeline_expected_n": pipeline_n,
        "pipeline_sponsor_entropy": rng.uniform(0, 3, n),
        "pipeline_design_het": rng.uniform(0, 1, n),
        "pipeline_empty": rng.choice([True, False], n),
    })


def _features_for_pico(effect, pl) -> dict:
    return dict(
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


def main() -> int:
    if not os.environ.get("TRUTHCERT_HMAC_KEY"):
        os.environ["TRUTHCERT_HMAC_KEY"] = "dev-mode-key-REPLACE-FOR-RELEASE"
        print("WARNING: using default dev HMAC key; set TRUTHCERT_HMAC_KEY for any non-dev run.")

    models_dir = ROOT / "models"
    outputs_dir = ROOT / "outputs"
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    print("[1/3] Training synthetic flip-forecaster...")
    df = _synthetic_training_df()
    artifacts = train_models(df, models_dir=models_dir)

    with open(artifacts.gbm_path, "rb") as f:
        bundle = pickle.load(f)
    report = validate_model(bundle, df, holdout_topic=None)
    write_validation_report(report, models_dir / "validation_report_v1.json")
    print(f"      synthetic AUC={report.auc:.3f} Brier={report.brier:.3f} n_test={report.n_test}")
    cardio_report = validate_model(bundle, df, holdout_topic="cardiology")
    write_validation_report(cardio_report, models_dir / "validation_report_v1_cardiology.json")
    print(f"      synthetic cardiology AUC={cardio_report.auc:.3f} n={cardio_report.n_test}")

    fixture_aact = ROOT / "tests" / "fixtures" / "aact_mini.csv"
    print(f"[2/3] Using fixture AACT at {fixture_aact}")

    print("[3/3] Generating trio cards...")
    native_pool = NativePoolBackend()
    for pico_name in ["sglt2i_hfpef", "tirzepatide_hfpef_acm", "empareg_t2dm"]:
        pico = load_pico(ROOT / "configs" / "picos" / f"{pico_name}.yaml")
        effect = compute_effect(pico, backend=native_pool)
        pl = extract_pipeline(pico, snapshot_date="2026-04-14", aact_path=fixture_aact)
        rep = compute_representativeness({}, {})
        features = _features_for_pico(effect, pl)
        flip = predict_flip(features, model_path=artifacts.gbm_path, bootstrap_n=200, seed=0)
        card = assemble_card(pico.id, effect, flip, rep)
        card["_dev_mode"] = {
            "status": "dev-mode artifact (native pool; synthetic flip model)",
            "effect_source": "native DL+HKSJ pool over hand-curated studies in configs/studies/*.studies.yaml",
            "model_training": "synthetic data — real MetaAudit version-pair dataset still blocked (see REAL_PHASE1_BLOCKERS.md)",
            "aact_source": "fixture subset — canonical extract integration pending",
            "representativeness": "empty — registry-first population layer integration pending",
            "bootstrap_utc": datetime.now(timezone.utc).isoformat(),
        }
        (outputs_dir / f"{pico.id}.json").write_text(json.dumps(card, indent=2))
        (outputs_dir / f"{pico.id}.html").write_text(render_html(card))
        print(f"      wrote {pico.id}.json (flip P={flip.probability:.3f}, "
              f"effect={effect.point:.2f}, pipeline_empty={pl.pipeline_empty})")

    _write_index_html(outputs_dir)
    print("Done. Outputs at", outputs_dir)
    return 0


def _write_index_html(outputs_dir: Path) -> None:
    cards = [
        ("sglt2i_hfpef", "SGLT2i in HFpEF"),
        ("tirzepatide_hfpef_acm", "Tirzepatide HFpEF ACM (SUMMIT)"),
        ("empareg_t2dm", "Empagliflozin T2DM CV outcomes (EMPA-REG)"),
    ]
    links = "\n".join(f'<li><a href="{pid}.html">{label}</a></li>' for pid, label in cards)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Evidence Forecast</title>
<style>body{{font-family:system-ui;max-width:640px;margin:3em auto;padding:0 1em;}}
.dev{{background:#fff3cd;border:1px solid #ffc107;padding:.8em;border-radius:6px;color:#6d4c00;}}
</style></head>
<body><h1>Evidence Forecast — Phase 1</h1>
<div class="dev"><strong>DEV MODE.</strong> Cards below are generated from stub effects anchored to published primary MAs
and a synthetic-data flip-forecaster. Real integration (CardioSynth + MetaAudit training) is the
user-present release step.</div>
<ul>{links}</ul></body></html>
"""
    (outputs_dir / "index.html").write_text(html)


if __name__ == "__main__":
    raise SystemExit(main())
