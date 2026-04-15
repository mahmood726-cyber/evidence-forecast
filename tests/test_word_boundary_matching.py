"""Contract: word-boundary AACT matching rejects short terms and substring hits.

These are regression guards for the parser-audit P0-7 finding: pre-fix,
substring matching made "hmg" match "hmg-coa reductase" (correct) but also
matched unrelated 3-char substrings inside longer drug names, producing
~25% false-positive contamination in the non-empty-pipeline set.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_per_pair_pipeline_features import extract_pipeline_fast


def _make_aact(iv: str, cd: str) -> pd.DataFrame:
    """Build a minimal AACT-shape dataframe with one ongoing trial."""
    from datetime import date
    return pd.DataFrame([{
        "interventions_lc": iv.lower(),
        "conditions_lc": cd.lower(),
        "overall_status": "RECRUITING",  # AACT v2 API uses uppercase
        "start_date_d": date(2020, 1, 1),
        "completion_date_d": date(2030, 1, 1),
        "enrollment": 100,
        "lead_sponsor": "Acme",
        "study_type": "INTERVENTIONAL",
        "phase": "PHASE3",
        "primary_purpose": "TREATMENT",
    }])


def test_short_intervention_term_rejected():
    """Heads < 4 chars are rejected outright to prevent contamination."""
    df = _make_aact("abc", "diabetes")
    pl = extract_pipeline_fast(df, "abc", "diabetes", "2025-01-01")
    assert pl.pipeline_empty, "3-char intervention should be rejected"
    assert pl.trial_count == 0


def test_short_condition_term_rejected():
    df = _make_aact("empagliflozin", "cad")
    pl = extract_pipeline_fast(df, "empagliflozin", "cad", "2025-01-01")
    assert pl.pipeline_empty, "3-char condition should be rejected"


def test_word_boundary_blocks_substring_false_positive():
    """'hmgb' (4-char) must not match 'hmgb1-antibody' when searched as 'hmgb'.
    Actually that WOULD match; the key regression is that 'hmgr' must not
    match 'mesmerised' where 'hmgr' occurs mid-word. Use a concrete case.
    """
    # 'stat' as an intervention term. In a world without word boundaries,
    # 'stat' would match 'tristatin', 'status', etc.
    df = _make_aact("tristatin 50mg", "cardiovascular disease")
    pl = extract_pipeline_fast(df, "stat", "cardiovascular", "2025-01-01")
    # 'stat' is 4 chars so not rejected for length; but word-boundary must
    # prevent matching inside 'tristatin'.
    # Actually 'stat' IS 4 chars. The rejection is < 4. So this specifically
    # tests the \b behaviour.
    assert pl.pipeline_empty, (
        "word-boundary regex must reject 'stat' matching 'tristatin' substring"
    )


def test_word_boundary_allows_real_word_match():
    df = _make_aact("empagliflozin 25mg daily", "type 2 diabetes mellitus")
    pl = extract_pipeline_fast(df, "empagliflozin", "diabetes", "2025-01-01")
    assert not pl.pipeline_empty
    assert pl.trial_count == 1


def test_empty_terms_return_empty_pipeline():
    df = _make_aact("empagliflozin", "diabetes")
    pl_iv_empty = extract_pipeline_fast(df, "", "diabetes", "2025-01-01")
    pl_cd_empty = extract_pipeline_fast(df, "empagliflozin", "", "2025-01-01")
    assert pl_iv_empty.pipeline_empty
    assert pl_cd_empty.pipeline_empty


def test_bad_snapshot_date_returns_empty():
    df = _make_aact("empagliflozin", "diabetes")
    pl = extract_pipeline_fast(df, "empagliflozin", "diabetes", "not-a-date")
    assert pl.pipeline_empty


def test_iso_snapshot_with_time_component_accepted():
    """P0-9 regression: pd.to_datetime must accept ISO with time suffix."""
    df = _make_aact("empagliflozin", "diabetes")
    pl = extract_pipeline_fast(df, "empagliflozin", "diabetes", "2025-01-01T00:00:00")
    assert not pl.pipeline_empty
