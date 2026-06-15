"""dgp_flip.py — STANDALONE seeded known-truth DGP for evidence-forecast.

Calibration test design
------------------------
The forecaster claims a PROBABILITY that a meta-analysis conclusion will flip
within the horizon. A probability is only meaningful if it is CALIBRATED:
among pairs the model assigns ~p, the realised flip rate should be ~p.

We construct a known truth: each synthetic MA "pair" has v1 features (the same
feature columns the engine trains on). The TRUE flip probability is a fixed
logistic function of a few of those features (effect geometry + heterogeneity +
evidence base size), so flips that are inherently unstable (wide CI, near-null
point, high tau2, small k) genuinely flip more often. We then draw the binary
flip ~ Bernoulli(true_p). This is the cumulative-evidence trajectory's terminal
outcome under a known generative law.

If the engine's trained model is well-behaved, its PREDICTED P(flip) on held-out
data should be calibrated against the realised flip rate (reliability slope ~1,
Brier low, per-bin predicted ~ observed). That is the truth-recovery measurement.

The DGP is fully seeded and independent of the package internals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# canonical feature columns the engine expects (must match FEATURE_COLS_V1)
FEATURES = [
    "ci_width", "pi_width", "distance_from_null",
    "k", "tau2", "i2",
    "fragility_index", "egger_p", "trim_fill_delta", "benford_mad",
    "pipeline_trial_count", "pipeline_expected_n",
    "pipeline_sponsor_entropy", "pipeline_design_het", "pipeline_empty",
    "v1_year_span", "v1_years_since_recent", "v1_annual_accrual",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_dataset(n=4000, seed=0):
    """Generate n synthetic MA pairs with a KNOWN logistic flip law.

    Returns a DataFrame with the engine's feature columns plus ma_id, flip,
    topic_area, v1_date, and a hidden `true_p` column (the generative truth).
    """
    rng = np.random.default_rng(seed)

    # --- v1 feature draws (plausible meta-analytic ranges) ---
    ci_width = rng.gamma(2.0, 0.25, n)                 # >0, typical CI widths on log scale
    distance_from_null = np.abs(rng.normal(0.0, 0.30, n))
    k = rng.integers(3, 40, n).astype(float)           # number of studies
    tau2 = rng.gamma(1.5, 0.05, n)                     # between-study variance
    i2 = np.clip(rng.beta(2, 3, n) * 100, 0, 100)      # %
    fragility_index = rng.poisson(6, n).astype(float)
    egger_p = rng.uniform(0, 1, n)
    trim_fill_delta = np.abs(rng.normal(0, 0.05, n))
    benford_mad = np.abs(rng.normal(0.01, 0.005, n))

    df = pd.DataFrame({
        "ci_width": ci_width,
        "pi_width": ci_width * rng.uniform(1.2, 2.5, n),
        "distance_from_null": distance_from_null,
        "k": k,
        "tau2": tau2,
        "i2": i2,
        "fragility_index": fragility_index,
        "egger_p": egger_p,
        "trim_fill_delta": trim_fill_delta,
        "benford_mad": benford_mad,
        "pipeline_trial_count": rng.poisson(8, n).astype(float),
        "pipeline_expected_n": rng.gamma(2, 1500, n),
        "pipeline_sponsor_entropy": rng.uniform(0, 2.5, n),
        "pipeline_design_het": rng.uniform(0, 1, n),
        "pipeline_empty": (rng.uniform(0, 1, n) < 0.15).astype(int),
        "v1_year_span": rng.uniform(1, 20, n),
        "v1_years_since_recent": rng.uniform(0, 10, n),
        "v1_annual_accrual": rng.gamma(2, 2, n),
    })

    # --- KNOWN generative flip law (the truth we recover) ---
    # A conclusion is more likely to flip when:
    #   - the CI is wide (unstable),         +
    #   - the point estimate is near null,   + (small distance_from_null)
    #   - heterogeneity (tau2) is high,      +
    #   - the evidence base is small (low k),+
    #   - the fragility index is low.        +
    # standardize the drivers so coefficients are interpretable
    z_ci   = (df["ci_width"] - df["ci_width"].mean()) / df["ci_width"].std()
    z_dist = (df["distance_from_null"] - df["distance_from_null"].mean()) / df["distance_from_null"].std()
    z_tau  = (df["tau2"] - df["tau2"].mean()) / df["tau2"].std()
    z_k    = (df["k"] - df["k"].mean()) / df["k"].std()
    z_frag = (df["fragility_index"] - df["fragility_index"].mean()) / df["fragility_index"].std()

    logit = (
        -1.2                      # base rate ~ flip prevalence
        + 0.9 * z_ci
        - 0.8 * z_dist
        + 0.7 * z_tau
        - 0.6 * z_k
        - 0.5 * z_frag
    )
    true_p = _sigmoid(logit.values)
    flip = (rng.uniform(0, 1, n) < true_p).astype(int)

    df["true_p"] = true_p
    df["flip"] = flip
    df["ma_id"] = [f"MA{i:05d}" for i in range(n)]
    df["topic_area"] = "cardiology"

    # temporal labels: spread v1 dates so the engine's temporal split has both sides
    years = rng.integers(2015, 2026, n)
    months = rng.integers(1, 13, n)
    df["v1_date"] = pd.to_datetime(
        [f"{y}-{m:02d}-15" for y, m in zip(years, months)]
    )
    return df
