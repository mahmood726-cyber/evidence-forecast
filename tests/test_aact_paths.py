"""Contract tests for the AACT path discovery helper.

Validates resolution order — CLI > env > project cache > candidate roots —
and fail-closed behaviour when no snapshot is locatable.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast._aact_paths import discover_root, discover_file


def test_discover_root_respects_cli_arg(tmp_path):
    p = discover_root(cli_root=str(tmp_path))
    assert p == tmp_path


def test_discover_root_respects_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("AACT_ROOT", str(tmp_path))
    p = discover_root()
    assert p == tmp_path


def test_discover_root_prefers_cli_over_env(monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("AACT_ROOT", str(other))
    p = discover_root(cli_root=str(tmp_path))
    assert p == tmp_path


def test_discover_root_fails_closed_when_nothing_found(monkeypatch):
    monkeypatch.delenv("AACT_ROOT", raising=False)
    # All candidate roots should be absent on CI / clean machines.
    # If one happens to exist we skip; if none do, we expect SystemExit.
    from evidence_forecast._aact_paths import _RAW_ROOT_CANDIDATES
    if any((Path(c) / "studies.txt").exists() for c in _RAW_ROOT_CANDIDATES):
        pytest.skip("at least one candidate root exists on this machine")
    with pytest.raises(SystemExit):
        discover_root()


def test_discover_file_respects_cli_arg(tmp_path):
    p = discover_file(cli_path=str(tmp_path / "foo.csv"))
    assert p == tmp_path / "foo.csv"


def test_discover_file_respects_env_var(monkeypatch, tmp_path):
    target = tmp_path / "foo.csv"
    monkeypatch.setenv("AACT_PATH", str(target))
    p = discover_file()
    assert p == target


def test_discover_file_prefers_project_cache_if_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("AACT_PATH", raising=False)
    cache = tmp_path / "aact_joined.csv"
    cache.write_text("cached")
    p = discover_file(project_cache=cache)
    assert p == cache


def test_discover_file_falls_back_to_cache_path_when_nothing_found(monkeypatch, tmp_path):
    monkeypatch.delenv("AACT_PATH", raising=False)
    # Cache file does not exist — helper must still return it (not one of the
    # global candidates) so callers can surface a "run build_aact_cache.py
    # first" message. To isolate from whatever lives on the dev machine we
    # patch the candidates tuple to point at paths guaranteed absent.
    import evidence_forecast._aact_paths as mod
    monkeypatch.setattr(mod, "_STUDIES_FILE_CANDIDATES",
                        (str(tmp_path / "_nope" / "studies.txt"),))
    cache = tmp_path / "aact_joined.csv"
    assert not cache.exists()
    p = discover_file(project_cache=cache)
    assert p == cache
