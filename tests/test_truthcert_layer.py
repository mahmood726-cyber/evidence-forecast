import json
import pytest
from evidence_forecast.truthcert_layer import (
    sign_bundle, verify_bundle, TruthCertError,
)
from evidence_forecast.constants import HMAC_ENV_VAR


def test_sign_and_verify_roundtrip(monkeypatch):
    monkeypatch.setenv(HMAC_ENV_VAR, "s" * 64)
    bundle = {"pico_id": "t", "effect": {"point": 0.8}}
    signed = sign_bundle(bundle)
    assert "truthcert" in signed
    assert signed["truthcert"]["sha256"]
    assert signed["truthcert"]["hmac_sha256"]
    assert verify_bundle(signed) is True


def test_tampered_bundle_fails_verification(monkeypatch):
    monkeypatch.setenv(HMAC_ENV_VAR, "s" * 64)
    bundle = {"pico_id": "t", "effect": {"point": 0.8}}
    signed = sign_bundle(bundle)
    signed["effect"]["point"] = 0.9  # tamper
    assert verify_bundle(signed) is False


def test_missing_hmac_env_fails_closed(monkeypatch):
    monkeypatch.delenv(HMAC_ENV_VAR, raising=False)
    with pytest.raises(TruthCertError) as exc:
        sign_bundle({"pico_id": "t"})
    assert "HMAC" in str(exc.value) or HMAC_ENV_VAR in str(exc.value)


def test_hmac_key_not_in_output(monkeypatch):
    key = "secret-key-must-not-leak"
    monkeypatch.setenv(HMAC_ENV_VAR, key)
    signed = sign_bundle({"pico_id": "t"})
    assert key not in json.dumps(signed)


def test_signature_uses_constant_time_compare(monkeypatch):
    """Regression: ensure verify uses hmac.compare_digest, not == ."""
    import inspect
    from evidence_forecast import truthcert_layer
    source = inspect.getsource(truthcert_layer.verify_bundle)
    assert "compare_digest" in source, "must use hmac.compare_digest"
