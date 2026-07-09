"""CLI: produce one Forecast Card for a named PICO.

Usage:
    python scripts/run_forecast.py --pico sglt2i_hfpef

Requires env:
    TRUTHCERT_HMAC_KEY  (any non-empty string for dev; real HMAC key for release)

AACT path resolution is delegated to evidence_forecast._aact_paths.discover_file
(honours --aact CLI, then AACT_PATH env, then project cache, then candidates).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast._aact_paths import discover_file as _discover_aact_file
from evidence_forecast._registry_first_paths import discover_root as _discover_registry_first_root
from evidence_forecast.pico_spec import load_pico
from evidence_forecast.effect_layer import compute_effect
from evidence_forecast.pipeline_layer import extract_pipeline
from evidence_forecast.representativeness import compute_representativeness
from evidence_forecast.flip_forecaster import predict_flip
from evidence_forecast.forecast_card import assemble_card, render_html


_REGISTRY_FIRST_REVIEW_IDS: dict[str, str] = {
    # Exact semantic match to registry-first topic config + population outputs.
    "sglt2i_hfpef": "hfpef_sglt2",
}

_TEMPORAL_FEATURE_DEFAULTS: dict[str, float] = {
    # Production PICOs do not carry v1 study-year distributions on disk.
    # Keep these aligned with the shipped trio generator.
    "v1_year_span": 10.0,
    "v1_years_since_recent": 0.0,
    "v1_annual_accrual": 0.833,
}


def _default_aact_path() -> str:
    return str(_discover_aact_file(
        project_cache=ROOT / "cache" / "aact_joined_2026-04-12.csv",
    ))


def _distance_from_null(point: float, scale: str) -> float:
    """Log-scale for ratio measures, natural-scale for differences (P0-4)."""
    if scale in {"HR", "OR", "RR"}:
        if point <= 0:
            return float("nan")
        return abs(math.log(point))
    return abs(point)


def _picos_dir() -> Path:
    return ROOT / "configs" / "picos"


def _available_picos() -> list[str]:
    return sorted(p.stem for p in _picos_dir().glob("*.yaml"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Produce one signed-style Forecast Card for a named PICO."
    )
    ap.add_argument("--pico", help="PICO id (see --list for available ids).")
    ap.add_argument("--list", action="store_true",
                    help="List available PICO ids and exit.")
    ap.add_argument("--aact", default=_default_aact_path(),
                    help="AACT joined CSV or raw studies.txt. "
                         "Defaults: AACT_PATH env, else local cache, else D: or C: candidates.")
    ap.add_argument("--model", default=str(ROOT / "models" / "flip_forecaster_v1.pkl"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--snapshot", default="2026-04-14")
    args = ap.parse_args()

    available = _available_picos()

    if args.list:
        print("Available PICOs:")
        for pid in available:
            print(f"  {pid}")
        return 0

    if not args.pico:
        ap.error("--pico is required (or use --list to see available PICO ids)")

    pico_path = _picos_dir() / f"{args.pico}.yaml"
    if not pico_path.exists():
        ap.error(
            f"unknown PICO {args.pico!r}. Available: {', '.join(available) or '(none found)'}. "
            "Use --list to see them."
        )

    pico = load_pico(pico_path)
    effect = compute_effect(pico)
    pl = extract_pipeline(pico, snapshot_date=args.snapshot, aact_path=Path(args.aact))
    tc_weights, burden = _load_weights(pico.id)
    rep_source = "registry_first" if tc_weights and burden else "none"
    rep = (compute_representativeness(tc_weights, burden, source=rep_source)
           if tc_weights and burden
           else compute_representativeness({}, {}))

    features = _build_forecaster_features(effect=effect, pl=pl)
    flip = predict_flip(features, model_path=Path(args.model))
    card = assemble_card(pico_id=pico.id, effect=effect, flip=flip, representativeness=rep)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{pico.id}.json").write_text(json.dumps(card, indent=2))
    (out_dir / f"{pico.id}.html").write_text(render_html(card))
    print(f"wrote {out_dir / (pico.id + '.json')} and {pico.id}.html")
    return 0


def _build_forecaster_features(*, effect, pl) -> dict[str, float | int | bool]:
    features: dict[str, float | int | bool] = dict(
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
    )
    features.update(_TEMPORAL_FEATURE_DEFAULTS)
    return features


def _load_weights(pico_id: str) -> tuple[dict, dict]:
    """Best-effort load from registry-first outputs; empty if absent.

    Fail-closed: only exact topic-aligned review outputs are used. If the
    sister project or the matching run directory is unavailable, return
    empty weights rather than guessing across mismatched conditions.
    """
    review_id = _REGISTRY_FIRST_REVIEW_IDS.get(pico_id)
    if not review_id:
        return {}, {}

    registry_root = _discover_registry_first_root()
    if registry_root is None:
        return {}, {}

    run_dir = registry_root / "outputs" / "population_runs" / review_id
    if not run_dir.exists():
        return {}, {}

    trial_country_weights = _read_weight_map(
        run_dir / "trial_country_weights.csv",
        value_fields=("enrollment_weight", "country_weight"),
    )
    burden_weights = _read_weight_map(
        run_dir / "country_rankings.csv",
        value_fields=("burden_number", "burden_value"),
    )
    if not burden_weights:
        burden_weights = _read_weight_map(
            run_dir / "burden_table.csv",
            value_fields=("burden_value",),
        )
    return trial_country_weights, burden_weights


def _read_weight_map(path: Path, *, value_fields: tuple[str, ...]) -> dict[str, float]:
    if not path.exists():
        return {}

    weights: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iso3c = str(row.get("iso3c") or "").strip().upper()
            if not iso3c:
                continue
            value = _parse_weight_value(row=row, value_fields=value_fields)
            if value is None:
                continue
            weights[iso3c] = weights.get(iso3c, 0.0) + value
    return _normalise_weights(weights)


def _parse_weight_value(*, row: dict[str, str], value_fields: tuple[str, ...]) -> float | None:
    for field in value_fields:
        raw = str(row.get(field) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        return {}
    return {
        iso3c: value / total
        for iso3c, value in sorted(weights.items())
        if math.isfinite(value) and value > 0
    }


if __name__ == "__main__":
    raise SystemExit(main())
