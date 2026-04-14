from pathlib import Path
import json
import pytest
from evidence_forecast.forecast_card import assemble_card, render_html
from evidence_forecast.effect_layer import EffectResult
from evidence_forecast.flip_forecaster import FlipForecast
from evidence_forecast.representativeness import RepresentativenessResult
from evidence_forecast.constants import CARD_TOP_LEVEL_FIELDS


@pytest.fixture
def parts():
    return dict(
        pico_id="sglt2i_hfpef",
        effect=EffectResult(0.79, 0.69, 0.91, 0.55, 1.13, 5, 0.012, 0.18, "HR"),
        flip=FlipForecast(0.12, 0.06, 0.22, 24, "1.0.0", False),
        representativeness=RepresentativenessResult(0.64, 18, True, "aact"),
    )


def test_assembled_card_has_all_top_level_fields(parts, seed_env):
    card = assemble_card(**parts)
    for f in CARD_TOP_LEVEL_FIELDS:
        assert f in card, f"missing {f}"


def test_card_is_json_serialisable(parts, seed_env):
    card = assemble_card(**parts)
    s = json.dumps(card)
    assert len(s) > 0


def test_card_signed_with_truthcert(parts, seed_env):
    card = assemble_card(**parts)
    assert "truthcert" in card
    assert card["truthcert"]["hmac_sha256"]
    assert card["truthcert"]["key_source"].startswith("env:")


def test_html_contains_pico_id_and_numbers(parts, seed_env):
    card = assemble_card(**parts)
    html = render_html(card)
    assert "sglt2i_hfpef" in html
    assert "0.79" in html
    assert "12%" in html or "0.12" in html
    assert "<html" in html.lower() and "</html>" in html.lower()


def test_html_includes_truthcert_hash(parts, seed_env):
    card = assemble_card(**parts)
    html = render_html(card)
    assert card["truthcert"]["sha256"][:16] in html
