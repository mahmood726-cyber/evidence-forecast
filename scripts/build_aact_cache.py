"""One-time builder: joins AACT raw tables into a single CSV matching the
schema that evidence_forecast.pipeline_layer.extract_pipeline expects.

AACT root discovery (per lessons.md — no hardcoded drive):
  1. --aact-root CLI flag
  2. AACT_ROOT environment variable
  3. Candidate roots: D:/AACT/2026-04-12, C:/Users/user/AACT/2026-04-12
  4. Fail closed if none found.

Columns emitted: nct_id, brief_title, overall_status, phase, study_type,
primary_purpose, enrollment, start_date, completion_date, lead_sponsor,
conditions, interventions.

Output is gitignored — this is a derived file reproducible from AACT raw.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


_CANDIDATE_ROOTS = (
    r"D:\AACT-storage\AACT\2026-04-12",
    r"D:\AACT\2026-04-12",
    r"C:\Users\user\AACT\2026-04-12",
)


def _discover_aact_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root)
    env = os.environ.get("AACT_ROOT")
    if env:
        return Path(env)
    for cand in _CANDIDATE_ROOTS:
        p = Path(cand)
        if (p / "studies.txt").exists():
            return p
    raise SystemExit(
        "AACT root not found. Set --aact-root or AACT_ROOT env var. "
        f"Searched: {_CANDIDATE_ROOTS}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aact-root", default=None,
                    help="AACT extract directory. Falls back to AACT_ROOT env, "
                         "then candidate roots (D: or C:), else fails closed.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = _discover_aact_root(args.aact_root)
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "cache" / "aact_joined_2026-04-12.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading studies from {root / 'studies.txt'}")
    studies = pd.read_csv(
        root / "studies.txt", sep="|", low_memory=False,
        usecols=["nct_id", "brief_title", "overall_status", "phase",
                 "study_type", "enrollment", "start_date", "completion_date"],
        dtype=str,
    )
    print(f"  {len(studies):,} trials")

    print("Reading designs (primary_purpose)")
    designs = pd.read_csv(
        root / "designs.txt", sep="|", low_memory=False,
        usecols=["nct_id", "primary_purpose"], dtype=str,
    )

    print("Reading sponsors (lead only)")
    sponsors = pd.read_csv(
        root / "sponsors.txt", sep="|", low_memory=False,
        usecols=["nct_id", "lead_or_collaborator", "name"], dtype=str,
    )
    lead_sponsors = (
        sponsors[sponsors["lead_or_collaborator"].str.lower() == "lead"]
        .drop_duplicates(subset=["nct_id"])
        .rename(columns={"name": "lead_sponsor"})[["nct_id", "lead_sponsor"]]
    )

    print("Reading interventions (aggregating per trial)")
    interventions = pd.read_csv(
        root / "interventions.txt", sep="|", low_memory=False,
        usecols=["nct_id", "name"], dtype=str,
    )
    iv_agg = (
        interventions.fillna("")
        .groupby("nct_id")["name"]
        .apply(lambda s: " | ".join(sorted(set(x for x in s if x))))
        .rename("interventions")
        .reset_index()
    )

    print("Reading conditions (aggregating per trial)")
    conditions = pd.read_csv(
        root / "conditions.txt", sep="|", low_memory=False,
        usecols=["nct_id", "name"], dtype=str,
    )
    cd_agg = (
        conditions.fillna("")
        .groupby("nct_id")["name"]
        .apply(lambda s: " | ".join(sorted(set(x for x in s if x))))
        .rename("conditions")
        .reset_index()
    )

    print("Joining")
    joined = (
        studies
        .merge(designs, on="nct_id", how="left")
        .merge(lead_sponsors, on="nct_id", how="left")
        .merge(iv_agg, on="nct_id", how="left")
        .merge(cd_agg, on="nct_id", how="left")
    )
    joined = joined[[
        "nct_id", "brief_title", "overall_status", "phase", "study_type",
        "primary_purpose", "enrollment", "start_date", "completion_date",
        "lead_sponsor", "conditions", "interventions",
    ]]
    print(f"  joined rows: {len(joined):,}")
    print(f"  writing {out}")
    joined.to_csv(out, index=False)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
