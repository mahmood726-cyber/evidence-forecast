import jsonschema
from evidence_forecast.forecast_card import assemble_card, CARD_JSON_SCHEMA
from evidence_forecast.effect_layer import EffectResult
from evidence_forecast.flip_forecaster import FlipForecast
from evidence_forecast.representativeness import RepresentativenessResult


def test_card_validates_against_schema(seed_env):
    card = assemble_card(
        pico_id="t",
        effect=EffectResult(0.8, 0.7, 0.9, 0.5, 1.1, 5, 0.01, 0.2, "HR"),
        flip=FlipForecast(0.1, 0.05, 0.2, 24, "1.0.0", False),
        representativeness=RepresentativenessResult(0.5, 10, True, "aact"),
    )
    jsonschema.validate(card, CARD_JSON_SCHEMA)
