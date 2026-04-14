from dataclasses import asdict
from evidence_forecast.effect_layer import EffectResult
from evidence_forecast.forecast_card import assemble_card, CARD_JSON_SCHEMA
from evidence_forecast.flip_forecaster import FlipForecast
from evidence_forecast.representativeness import RepresentativenessResult
import jsonschema


def test_effect_result_field_names_match_card_effect_schema(seed_env):
    e = EffectResult(0.8, 0.7, 0.9, 0.5, 1.1, 5, 0.01, 0.2, "HR")
    card = assemble_card(
        pico_id="x", effect=e,
        flip=FlipForecast(0.1, 0.05, 0.2, 24, "1.0.0", False),
        representativeness=RepresentativenessResult(0.5, 10, True, "aact"),
    )
    # None of the effect fields should have been dropped or silently renamed
    assert set(card["effect"].keys()) == set(asdict(e).keys())
    jsonschema.validate(card, CARD_JSON_SCHEMA)
