"""TruthCert layer: HMAC-SHA256 signature over canonicalised JSON.

Per lessons.md:
- HMAC key must come from env var TRUTHCERT_HMAC_KEY, never from the bundle.
- Constant-time comparison only (hmac.compare_digest).
- No placeholder signatures; fail closed when key is missing.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any

from evidence_forecast.constants import HMAC_ENV_VAR


class TruthCertError(RuntimeError):
    """Raised when TruthCert signing or verification cannot proceed."""


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _get_key() -> bytes:
    key = os.environ.get(HMAC_ENV_VAR, "")
    if not key:
        raise TruthCertError(
            f"HMAC key missing: env var {HMAC_ENV_VAR} must be set. "
            "TruthCert refuses to sign without an out-of-bundle key."
        )
    return key.encode("utf-8")


def sign_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    if "truthcert" in bundle:
        raise TruthCertError("bundle already contains 'truthcert' field; refusing to overwrite")
    key = _get_key()
    payload = _canonical_json(bundle)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    mac = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    signed = dict(bundle)
    signed["truthcert"] = {
        "sha256": sha,
        "hmac_sha256": mac,
        "signed_utc": datetime.now(timezone.utc).isoformat(),
        "key_source": f"env:{HMAC_ENV_VAR}",
    }
    return signed


def verify_bundle(signed: dict[str, Any]) -> bool:
    if "truthcert" not in signed:
        return False
    tc = signed["truthcert"]
    payload_bundle = {k: v for k, v in signed.items() if k != "truthcert"}
    payload = _canonical_json(payload_bundle)
    key = _get_key()
    expected_mac = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    expected_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (
        hmac.compare_digest(expected_mac, tc.get("hmac_sha256", ""))
        and hmac.compare_digest(expected_sha, tc.get("sha256", ""))
    )
