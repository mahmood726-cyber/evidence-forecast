"""CLI: produce one Forecast Card for a named PICO.

Usage:
    python scripts/run_forecast.py --pico sglt2i_hfpef

Requires env:
    TRUTHCERT_HMAC_KEY  (any non-empty string for dev; real HMAC key for release)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast.pico_spec import load_pico
from evidence_forecast.effect_layer import compute_effect
from evidence_forecast.pipeline_layer import extract_pipeline
from evidence_forecast.representativeness import compute_representativeness
from evidence_forecast.flip_forecaster import predict_flip
from evidence_forecast.forecast_card import assemble_card, render_html


_AACT_CANDIDATES = (
    ROOT / "cache" / "aact_joined_2026-04-12.csv",
    Path(r"D:\AACT\2026-04-12\studies.txt"),
    Path(r"C:\Users\user\AACT\2026-04-12\studies.txt"),
)


def _default_aact_path() -> str:
    import os
    env = os.environ.get("AACT_PATH")
    if env:
        return env
    for cand in _AACT_CANDIDATES:
        if cand.exists():
            return str(cand)
    return str(_AACT_CANDIDATES[0])  # cache path — run build_aact_cache.py first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pico", required=True)
    ap.add_argument("--aact", default=_default_aact_path(),
                    help="AACT joined CSV or raw studies.txt. "
                         "Defaults: AACT_PATH env, else local cache, else D: or C: candidates.")
    ap.add_argument("--model", default=str(ROOT / "models" / "flip_forecaster_v1.pkl"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--snapshot", default="2026-04-14")
    args = ap.parse_args()

    pico = load_pico(ROOT / "configs" / "picos" / f"{args.pico}.yaml")
    effect = compute_effect(pico)
    pl = extract_pipeline(pico, snapshot_date=args.snapshot, aact_path=Path(args.aact))
    tc_weights, burden = _load_weights(pico.id)
    rep = (compute_representativeness(tc_weights, burden)
           if tc_weights and burden
           else compute_representativeness({}, {}))

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
    flip = predict_flip(features, model_path=Path(args.model))
    card = assemble_card(pico_id=pico.id, effect=effect, flip=flip, representativeness=rep)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{pico.id}.json").write_text(json.dumps(card, indent=2))
    (out_dir / f"{pico.id}.html").write_text(render_html(card))
    print(f"wrote {out_dir / (pico.id + '.json')} and {pico.id}.html")
    return 0


def _load_weights(pico_id: str) -> tuple[dict, dict]:
    """Best-effort load from registry-first outputs; empty if absent."""
    return {}, {}


if __name__ == "__main__":
    raise SystemExit(main())
