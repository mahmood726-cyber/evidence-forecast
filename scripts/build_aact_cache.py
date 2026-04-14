"""One-time builder: joins AACT raw tables into a single CSV matching the
schema that evidence_forecast.pipeline_layer.extract_pipeline expects.

Input:  C:\\Users\\user\\AACT\\2026-04-12\\  (pipe-delimited .txt files)
Output: cache/aact_joined_2026-04-12.csv   (comma-delimited, one row per trial)

Columns emitted: nct_id, brief_title, overall_status, phase, study_type,
primary_purpose, enrollment, start_date, completion_date, lead_sponsor,
conditions, interventions.

Output is gitignored — this is a derived file reproducible from AACT raw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aact-root", default=r"C:\Users\user\AACT\2026-04-12")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.aact_root)
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
