"""v1-snapshot feature builder for flip-forecaster calibration.

No v2-column leakage: every feature name is strictly v1-sourced or
derived from AACT-at-v1-date.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from evidence_forecast.calibration.label_flips import label_flips
from evidence_forecast.pipeline_layer import extract_pipeline
from evidence_forecast.pico_spec import PICO


def build_features(pairs_path: Path, aact_path: Path) -> pd.DataFrame:
    labelled = label_flips(pairs_path)
    raw = pd.read_csv(pairs_path)
    merged = labelled.merge(
        raw.drop(columns=[c for c in labelled.columns if c in raw.columns and c != "ma_id"]),
        on="ma_id", how="left"
    )

    # Effect geometry
    merged["ci_width"] = merged["v1_ci_high"] - merged["v1_ci_low"]
    nulls = merged["scale"].map({"HR": 1.0, "OR": 1.0, "RR": 1.0,
                                 "RD": 0.0, "MD": 0.0, "SMD": 0.0})
    merged["distance_from_null"] = (merged["v1_point"] - nulls).abs()
    merged["pi_width"] = np.nan  # filled if source has PI; else NaN

    # Rename v1_* to canonical feature names
    rename = {
        "v1_k": "k",
        "v1_tau2": "tau2",
        "v1_i2": "i2",
        "v1_fragility_index": "fragility_index",
        "v1_egger_p": "egger_p",
        "v1_trim_fill_delta": "trim_fill_delta",
        "v1_benford_mad": "benford_mad",
    }
    for src, dst in rename.items():
        if src in merged.columns:
            merged[dst] = merged[src]

    # Pipeline features per v1 snapshot
    pipeline_rows = [_pipeline_for_row(r, aact_path) for _, r in merged.iterrows()]
    pipeline_df = pd.DataFrame(pipeline_rows)
    out = pd.concat([merged.reset_index(drop=True), pipeline_df], axis=1)

    keep = [
        "ma_id", "flip", "topic_area",
        "ci_width", "pi_width", "distance_from_null",
        "k", "tau2", "i2",
        "fragility_index", "egger_p", "trim_fill_delta",
        "benford_mad",
        "pipeline_trial_count", "pipeline_expected_n",
        "pipeline_sponsor_entropy", "pipeline_design_het",
        "pipeline_empty",
    ]
    return out[keep].copy()


def _pipeline_for_row(row: pd.Series, aact_path: Path) -> dict:
    pico = PICO(
        id=str(row["ma_id"]), title="",
        population=row.get("v1_population_term", ""),
        intervention=row.get("v1_intervention_term", ""),
        comparator="", outcome=str(row["outcome"]),
        outcome_type="binary", decision_threshold=1.0,
    )
    feats = extract_pipeline(pico, snapshot_date=str(row["v1_date"])[:10], aact_path=aact_path)
    return dict(
        pipeline_trial_count=feats.trial_count,
        pipeline_expected_n=feats.expected_n_sum,
        pipeline_sponsor_entropy=feats.sponsor_entropy,
        pipeline_design_het=feats.design_heterogeneity,
        pipeline_empty=feats.pipeline_empty,
    )
