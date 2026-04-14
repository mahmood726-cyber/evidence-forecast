"""Forecast Card assembler: combines four layers into a signed JSON bundle."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from evidence_forecast.constants import SCHEMA_VERSION
from evidence_forecast.effect_layer import EffectResult
from evidence_forecast.flip_forecaster import FlipForecast
from evidence_forecast.representativeness import RepresentativenessResult
from evidence_forecast.truthcert_layer import sign_bundle


CARD_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "pico_id", "generated_utc",
                 "effect", "flip", "representativeness", "truthcert"],
    "properties": {
        "schema_version": {"type": "string"},
        "pico_id": {"type": "string"},
        "generated_utc": {"type": "string"},
        "effect": {"type": "object", "required": ["point", "ci_low", "ci_high"]},
        "flip": {"type": "object", "required": ["probability", "ci_low", "ci_high", "horizon_months"]},
        "representativeness": {"type": "object", "required": ["overlap_score"]},
        "truthcert": {"type": "object", "required": ["sha256", "hmac_sha256", "signed_utc", "key_source"]},
    },
}


_TEMPLATE_DIR = Path(__file__).parent / "templates"


def assemble_card(
    pico_id: str,
    effect: EffectResult,
    flip: FlipForecast,
    representativeness: RepresentativenessResult,
) -> dict:
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "pico_id": pico_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "effect": asdict(effect),
        "flip": asdict(flip),
        "representativeness": asdict(representativeness),
    }
    return sign_bundle(bundle)


def render_html(card: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
    )
    tpl = env.get_template("card.html.j2")
    return tpl.render(card=card)
