"""Enrich the temporal pair CSV with real AACT pipeline features per pair.

For each training pair:
  1. Look up the review's Cochrane title (fetched via CrossRef).
  2. Parse the title into (intervention, condition) match terms using the
     dominant "X for Y" pattern with known-prefix/versus handling.
  3. Query the AACT joined cache at v1_date with those terms.
  4. Emit an enriched pair row with real pipeline_* features.

Output: cache/temporal_cochrane_pairs_enriched.csv

Usage:
    python scripts/build_per_pair_pipeline_features.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import math
from datetime import date
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_forecast.pipeline_layer import (
    PipelineFeatures, _ONGOING_STATUSES, _primary_token,
    _shannon_entropy, _design_heterogeneity,
)


def extract_pipeline_fast(
    df: pd.DataFrame,
    iv_term: str,
    cd_term: str,
    snapshot_date: str,
) -> PipelineFeatures:
    """In-memory variant of extract_pipeline: DataFrame already loaded once."""
    if not iv_term or not cd_term:
        return PipelineFeatures(0, 0, 0.0, 0.0, 0.0, True)
    snap = date.fromisoformat(snapshot_date)
    mask = (
        df["interventions_lc"].str.contains(iv_term.lower(), na=False, regex=False)
        & df["conditions_lc"].str.contains(cd_term.lower(), na=False, regex=False)
        & df["overall_status"].isin(_ONGOING_STATUSES)
        & (df["start_date_d"] <= snap)
        & (df["completion_date_d"] > snap)
    )
    matches = df[mask]
    if matches.empty:
        return PipelineFeatures(0, 0, 0.0, 0.0, 0.0, True)
    trial_count = len(matches)
    n_sum = int(matches["enrollment"].fillna(0).sum())
    sponsor_entropy = _shannon_entropy(matches["lead_sponsor"].fillna("UNKNOWN").tolist())
    design_heterogeneity = _design_heterogeneity(matches)
    return PipelineFeatures(
        trial_count=trial_count,
        expected_n_sum=n_sum,
        mean_expected_event_rate=0.05,
        sponsor_entropy=sponsor_entropy,
        design_heterogeneity=design_heterogeneity,
        pipeline_empty=False,
    )


PAIRS_IN = ROOT / "tests" / "fixtures" / "temporal_cochrane_pairs_v0.csv"
TITLES_IN = ROOT / "cache" / "cochrane_titles.csv"
AACT_CACHE = ROOT / "cache" / "aact_joined_2026-04-12.csv"
ENRICHED_OUT = ROOT / "cache" / "temporal_cochrane_pairs_enriched.csv"

# Prefixes to strip from the intervention side before taking head words.
_PREFIX_STRIP = [
    "effects of ", "effect of ", "use of ", "role of ",
    "impact of ", "evaluation of ", "training health professionals in ",
    "strategies for ", "interventions for ", "screening for ",
    "preoperative ", "postoperative ",
]

# Common outcome phrases that should NOT drive the condition split.
_IGNORE_IN_CONDITION = [
    "preventing ", "prevention of ", "treatment of ", "maintenance of ",
    "management of ", "reducing ", "reduction of ",
    "improving ", "risk of ",
]


def parse_pico(title: str) -> tuple[str, str]:
    """Return (intervention_term, condition_term) from a Cochrane title.

    Dominant Cochrane pattern: "<intervention> for <condition>".
    Comparison pattern: "<X> versus <Y> for <condition>".
    """
    if not isinstance(title, str) or not title.strip():
        return "", ""
    t = title.strip().lower()
    for p in _PREFIX_STRIP:
        if t.startswith(p):
            t = t[len(p):]
            break

    # Comparison: take first term as intervention.
    if " versus " in t:
        head, rest = t.split(" versus ", 1)
        iv = head.strip()
        cd = rest.split(" for ", 1)[1].strip() if " for " in rest else rest.strip()
    elif " for " in t:
        head, tail = t.split(" for ", 1)
        iv = head.strip()
        cd = tail.strip()
    elif " in " in t:
        head, tail = t.split(" in ", 1)
        iv = head.strip()
        cd = tail.strip()
    else:
        iv, cd = t, t

    # Strip outcome-like prefixes from condition.
    for p in _IGNORE_IN_CONDITION:
        if cd.startswith(p):
            cd = cd[len(p):]
            break

    # Take the first content word (≥3 chars, alphabetic) as the AACT search term.
    def _head_word(s: str) -> str:
        for tok in s.split():
            cleaned = re.sub(r"[^a-z]", "", tok)
            if len(cleaned) >= 3:
                return cleaned
        return s.split()[0] if s.split() else ""

    return _head_word(iv), _head_word(cd)


def main() -> int:
    if not TITLES_IN.exists():
        print(f"missing {TITLES_IN}; run fetch_cochrane_titles.py first")
        return 1
    if not AACT_CACHE.exists():
        print(f"missing {AACT_CACHE}; run build_aact_cache.py first")
        return 1

    titles_df = pd.read_csv(TITLES_IN)
    title_by_review = {}
    for _, r in titles_df.iterrows():
        doi = r["doi"]
        m = re.search(r"14651858\.(CD\d+)\.", doi)
        if m and r["title"]:
            title_by_review[m.group(1)] = r["title"]
    print(f"indexed {len(title_by_review)} review->title mappings")

    pairs = pd.read_csv(PAIRS_IN)
    review_ids = pairs["ma_id"].str.extract(r"(CD\d+)")[0]
    n_with_title = review_ids.isin(title_by_review).sum()
    print(f"{n_with_title} / {len(pairs)} pairs have matched titles")

    print("loading AACT cache once (579k rows)...", flush=True)
    (ROOT / "cache" / "_enrich_progress.txt").write_text("loading AACT cache")
    aact = pd.read_csv(AACT_CACHE, low_memory=False)
    print(f"  CSV read: {len(aact):,} rows", flush=True)
    aact["interventions_lc"] = aact["interventions"].fillna("").str.lower()
    aact["conditions_lc"] = aact["conditions"].fillna("").str.lower()
    print("  lower-cased text columns", flush=True)
    aact["start_date_d"] = pd.to_datetime(aact["start_date"], errors="coerce").dt.date
    aact["completion_date_d"] = pd.to_datetime(aact["completion_date"], errors="coerce").dt.date
    print(f"  ready: {len(aact):,} trial rows", flush=True)
    (ROOT / "cache" / "_enrich_progress.txt").write_text(f"AACT loaded {len(aact):,}")

    enriched_rows = []
    for idx, row in pairs.iterrows():
        review_id = re.match(r"(CD\d+)", row["ma_id"]).group(1)
        title = title_by_review.get(review_id, "")
        iv_term, cd_term = parse_pico(title) if title else ("", "")

        feats = extract_pipeline_fast(aact, iv_term, cd_term, row["v1_date"])

        out = dict(row)
        out["review_title"] = title
        out["match_intervention"] = iv_term
        out["match_condition"] = cd_term
        out["pipeline_trial_count"] = feats.trial_count
        out["pipeline_expected_n"] = feats.expected_n_sum
        out["pipeline_sponsor_entropy"] = feats.sponsor_entropy
        out["pipeline_design_het"] = feats.design_heterogeneity
        out["pipeline_empty"] = feats.pipeline_empty
        enriched_rows.append(out)

        if (idx + 1) % 200 == 0:
            msg = f"  processed {idx + 1}/{len(pairs)}"
            print(msg, flush=True)
            (ROOT / "cache" / "_enrich_progress.txt").write_text(msg)

    out_df = pd.DataFrame(enriched_rows)
    ENRICHED_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(ENRICHED_OUT, index=False)
    non_empty = (out_df["pipeline_empty"] == False).sum()
    print(f"\nenriched {len(out_df)} pairs; {non_empty} ({non_empty/len(out_df)*100:.1f}%) "
          f"have non-empty AACT pipeline")
    print(f"mean pipeline_trial_count (non-empty): "
          f"{out_df.loc[out_df['pipeline_empty']==False, 'pipeline_trial_count'].mean():.1f}")
    print(f"wrote {ENRICHED_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
