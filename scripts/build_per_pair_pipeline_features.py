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
    """In-memory variant of extract_pipeline: DataFrame already loaded once.

    Uses word-boundary regex (``\\b{term}\\b``) for AACT field matching so that
    short heads (e.g., ``hmg``, ``cad``) do not create spurious substring hits
    in unrelated trials (review-findings P0-7). Terms shorter than 4 chars are
    rejected outright as too ambiguous for substring intervention matching.
    """
    if not iv_term or not cd_term:
        return PipelineFeatures(0, 0, 0.0, 0.0, 0.0, True)
    if len(iv_term) < 4 or len(cd_term) < 4:
        return PipelineFeatures(0, 0, 0.0, 0.0, 0.0, True)
    snap_ts = pd.to_datetime(snapshot_date, errors="coerce")
    if pd.isna(snap_ts):
        return PipelineFeatures(0, 0, 0.0, 0.0, 0.0, True)
    snap = snap_ts.date()
    iv_pat = rf"\b{re.escape(iv_term.lower())}\b"
    cd_pat = rf"\b{re.escape(cd_term.lower())}\b"
    mask = (
        df["interventions_lc"].str.contains(iv_pat, na=False, regex=True)
        & df["conditions_lc"].str.contains(cd_pat, na=False, regex=True)
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

# Prefixes to strip from the intervention side before splitting.
_PREFIX_STRIP = [
    "effects of ", "effect of ", "use of ", "role of ",
    "impact of ", "evaluation of ", "training health professionals in ",
    "strategies for ", "screening for ",
    "preoperative ", "postoperative ",
    # Generic "interventions/treatments/...": strip so the next words drive iv.
    "pharmacological interventions for ", "non-pharmacological interventions for ",
    "interventions for ", "interventions to ", "interventions in ",
    "pharmacological treatments for ", "non-pharmacological treatments for ",
    "treatments for ", "treatment for ",
    "any intervention for ", "behavioural interventions for ",
    "psychosocial interventions for ", "educational interventions for ",
    "complementary therapies for ",
]

# Verb-prefixes inside the condition slot that should be stripped so the
# condition substring is the disease/outcome, not the action verb.
_CONDITION_VERB_PREFIX = [
    "preventing ", "prevention of ", "treating ", "treatment of ",
    "maintenance of ", "management of ", "managing ",
    "reducing ", "reduction of ", "reduction in ",
    "improving ", "improvement of ", "improvement in ",
    "risk of ",
    "the prevention of ", "the treatment of ", "the management of ",
    "the reduction of ", "the improvement of ", "the risk of ",
    "primary prevention of ", "secondary prevention of ",
    "the primary prevention of ", "the secondary prevention of ",
    "post-operative ", "postoperative ", "perioperative ", "preoperative ",
    "the postoperative ", "the perioperative ",
]

# Verb-prefixes that follow PREFIX_STRIP and indicate the condition is what
# follows the verb (no real intervention extractable).
_VERB_ONLY_OPENERS = (
    "prevent ", "treat ", "reduce ", "improve ", "manage ", "detect ",
    "preventing ", "treating ", "reducing ", "improving ", "managing ",
    "screen for ", "screening for ",
)

# Tokens that MUST NOT be returned as the condition head-word — they are
# articles, demographics, generic non-disease modifiers, or methodology.
_HEAD_WORD_BLACKLIST = {
    "the", "and", "with", "for", "from", "into", "onto", "upon",
    "treating", "preventing", "improving", "reducing", "managing",
    "older", "younger", "young", "adult", "adults", "child", "children",
    "infant", "infants", "neonate", "neonates", "person", "persons",
    "people", "patient", "patients", "women", "men", "male", "female",
    "primary", "secondary", "tertiary", "general", "standard",
    "early", "late", "acute", "chronic",
}

_DEMOGRAPHIC_MARKERS = (
    " in older ", " in young ", " in adult ", " in children ", " in infants ",
    " in neonates ", " in women ", " in men ", " in pregnant ",
    " in patient", " in people ",
)


def parse_pico(title: str) -> tuple[str, str]:
    """Return (intervention_term, condition_term) from a Cochrane title.

    Cochrane title patterns handled:
      "<iv> for <cd>"             — dominant pattern
      "<iv> versus <iv2> for <cd>" — head-to-head comparison
      "<iv> compared with <iv2> [for <cd>]" — alternate comparison phrasing
      "<iv> to prevent/treat/reduce <cd>" — purpose phrasing
      "<iv> in <cd>"              — fallback "in" split (often demographic)
      "<iv> following/after/during <cd>" — temporal phrasing
    Generic-noun openers ("interventions for", "treatments for", "pharmacological
    interventions for", ...) are stripped via ``_PREFIX_STRIP``.
    Article/verb-prefix leakage on the condition side ("for the prevention of X",
    "for treating Y") is stripped via ``_CONDITION_VERB_PREFIX``.
    """
    if not isinstance(title, str) or not title.strip():
        return "", ""
    t = title.strip().lower()
    # Drop trailing subtitle (": a network meta-analysis").
    if ":" in t:
        t = t.split(":", 1)[0].strip()
    # Drop parentheticals to reduce noise ("(NOACs)", "(HPV)" etc.).
    t = re.sub(r"\s*\([^)]*\)", "", t)
    prefix_stripped = False
    for p in _PREFIX_STRIP:
        if t.startswith(p):
            t = t[len(p):]
            prefix_stripped = True
            break

    # If a generic-noun opener was stripped and the next words are a verb
    # ("prevent X", "treat Y"), the iv slot has no real intervention; treat
    # the post-verb tail as condition.
    if prefix_stripped:
        for v in _VERB_ONLY_OPENERS:
            if t.startswith(v):
                return "", _term(_strip_condition_prefix(t[len(v):].strip()))

    # Handle "<A> compared with <B>" similarly to "versus".
    for compare_kw in (" versus ", " compared with ", " compared to ", " vs ", " vs. "):
        if compare_kw in t:
            head, rest = t.split(compare_kw, 1)
            iv = head.strip()
            for sep in (" for ", " in "):
                if sep in rest:
                    cd = rest.split(sep, 1)[1].strip()
                    break
            else:
                cd = rest.strip()
            return _term(iv), _term(_strip_condition_prefix(cd))

    # Purpose phrasings.
    for purpose_kw in (
        " to prevent ", " to treat ", " to reduce ", " to improve ",
        " to manage ", " to detect ",
    ):
        if purpose_kw in t:
            head, tail = t.split(purpose_kw, 1)
            return _term(head.strip()), _term(tail.strip())

    # If a " for " split is available, prefer it to temporal keywords —
    # "X commenced before Y for preventing Z" is a "for"-grammar title, not
    # a temporal one.
    if " for " in t:
        head, tail = t.split(" for ", 1)
        return _term(head.strip()), _term(_strip_condition_prefix(tail.strip()))

    # Temporal phrasings: "<iv> following/after/during/before <cd>".
    for temporal_kw in (" following ", " after ", " during ", " before "):
        if temporal_kw in t:
            head, tail = t.split(temporal_kw, 1)
            return _term(head.strip()), _term(tail.strip())

    if " in " in t:
        head, tail = t.split(" in ", 1)
        return _term(head.strip()), _term(_strip_condition_prefix(tail.strip()))

    return _term(t), _term(t)


def _strip_condition_prefix(cd: str) -> str:
    """Strip article + verb-prefix leakage from the condition side."""
    for _ in range(3):  # may need to chain (e.g., "the prevention of the")
        for p in _CONDITION_VERB_PREFIX:
            if cd.startswith(p):
                cd = cd[len(p):]
                break
        else:
            break
    return cd


def _term(s: str) -> str:
    """Pick the most specific 1–2 word AACT search term, skipping blacklist."""
    if not s:
        return ""
    tokens = []
    for tok in s.split():
        cleaned = re.sub(r"[^a-z]", "", tok)
        if len(cleaned) >= 3 and cleaned not in _HEAD_WORD_BLACKLIST:
            tokens.append(cleaned)
            if len(tokens) >= 1:
                break
    return tokens[0] if tokens else ""


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
    skipped_no_review_id = 0
    for idx, row in pairs.iterrows():
        m = re.match(r"(CD\d+)", str(row["ma_id"]))
        if not m:
            skipped_no_review_id += 1
            review_id = ""
        else:
            review_id = m.group(1)
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
    if skipped_no_review_id:
        print(f"WARNING: {skipped_no_review_id} pairs had ma_id without CDnnn prefix")
    non_empty = (~out_df["pipeline_empty"].astype(bool)).sum()
    print(f"\nenriched {len(out_df)} pairs; {non_empty} ({non_empty/len(out_df)*100:.1f}%) "
          f"have non-empty AACT pipeline")
    print(f"mean pipeline_trial_count (non-empty): "
          f"{out_df.loc[out_df['pipeline_empty']==False, 'pipeline_trial_count'].mean():.1f}")
    print(f"wrote {ENRICHED_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
