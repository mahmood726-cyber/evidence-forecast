"""Reject malformed certificates and non-JSON numbers, using software fixtures."""
import hashlib
import hmac
import json

import pytest

from evidence_forecast.constants import HMAC_ENV_VAR
from evidence_forecast.truthcert_layer import TruthCertError, sign_bundle, verify_bundle


@pytest.fixture(autouse=True)
def signing_key(monkeypatch):
    monkeypatch.setenv(HMAC_ENV_VAR, "regression-test-key")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_signing_rejects_non_finite_payload(value):
    with pytest.raises(TruthCertError, match="JSON"):
        sign_bundle({"effect": {"values": [value]}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_legacy_non_finite_signature_is_not_accepted(value):
    bundle = {"effect": {"point": value}}
    payload = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    bundle["truthcert"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hmac_sha256": hmac.new(b"regression-test-key", payload, hashlib.sha256).hexdigest(),
    }
    assert verify_bundle(bundle) is False


@pytest.mark.parametrize("certificate", [None, [], "invalid", 4])
def test_malformed_certificate_returns_false(certificate):
    assert verify_bundle({"effect": {}, "truthcert": certificate}) is False


@pytest.mark.parametrize("field", ["sha256", "hmac_sha256"])
@pytest.mark.parametrize("value", [None, 42, [], "é" * 64])
def test_malformed_digest_returns_false(field, value):
    signed = sign_bundle({"effect": {"point": 0.8}})
    signed["truthcert"][field] = value
    assert verify_bundle(signed) is False


def test_valid_signature_survives_json_roundtrip():
    signed = sign_bundle({"label": "software fixture α", "values": [0.0, None, True]})
    assert verify_bundle(json.loads(json.dumps(signed, allow_nan=False))) is True


def test_missing_key_still_raises_for_valid_certificate(monkeypatch):
    signed = sign_bundle({"effect": {"point": 0.8}})
    monkeypatch.delenv(HMAC_ENV_VAR)
    with pytest.raises(TruthCertError, match="HMAC key missing"):
        verify_bundle(signed)
