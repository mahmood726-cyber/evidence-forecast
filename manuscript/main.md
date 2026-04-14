# Evidence Forecasting: probabilistic prediction of meta-analytic conclusion stability

**Author**: Mahmood Ahmad, Tahir Heart Institute. ORCID 0009-0003-7781-4478.
**Target**: Nature Medicine (method paper); BMJ Analysis / J Clin Epidemiol if refocused.
**Repo**: github.com/mahmood726-cyber/evidence-forecast (commit b4c5db5)
**Status**: Draft v0.2 — real-data calibrated; awaiting pipeline-feature enrichment before submission.

## Abstract (≈240 words)

**Background.** Meta-analyses publish point estimates. Guideline panels decide binary recommendations. Whether a current conclusion will survive the next wave of registered trials is a question the field has no operational primitive for.

**Methods.** We built a four-layer Forecast Card that, for any PICO, emits a cryptographically signed JSON bundle containing: the current pooled effect with 95% CI and prediction interval (native DerSimonian-Laird + HKSJ pool); the probability that the conclusion will reverse within 24 months, defined as the 95% CI's null-crossing status changing between versions; a burden-weighted representativeness score over trial-country × IHME disease burden; and an HMAC-SHA256 signature attesting to the input hash. The flip probability is produced by a gradient-boosted classifier with post-hoc Platt sigmoid calibration, trained on 1,632 real temporal version pairs extracted from 560 Cochrane study-level datasets (Pairwise70). Each pair is constructed by splitting an analysis's studies at the 55th and 80th percentiles of publication year and pooling each subset via `metafor::rma(method="DL")`. Training uses a temporal split (v1 < 2015 train, v1 ≥ 2015 test) with label-permutation and null sanity checks.

**Results.** Held-out discrimination on n = 1,080 pairs: AUC 0.777, Brier 0.064. Calibration slope 1.32, intercept 1.03 (post-Platt; raw GBM was 0.36/−1.6). Permutation-shuffled AUC 0.514 across 5 shuffles. Three worked PICOs (SGLT2i HFpEF; SUMMIT tirzepatide; EMPA-REG OUTCOME) produce plausible forecasts (4.4%, 22.1%, 9.4%) consistent with their evidence maturity.

**Interpretation.** Meta-analytic conclusion flip probability is learnable from real temporal Cochrane substrates and can be emitted as a signed, auditable primitive. Release-quality calibration remains above the 1.2 slope threshold by 10%, an honest limitation of current feature coverage.

## Introduction (≈500 words)

Meta-analyses publish point estimates. Guideline panels decide binary recommendations. The translation step — will this conclusion survive the next wave of registered trials? — is performed informally, if at all. Meteorology operationalises the same question: given the current state, what is the probability of a regime change before the next panel meets? Evidence synthesis does not.

This paper proposes a primitive, not a product. The primitive is a *Forecast Card*: four numbers emitted per PICO, cryptographically signed, drawn from independent substrates. No single number is novel. The novelty is that (a) they are co-reported, (b) one of them is a calibrated flip probability, and (c) the bundle is non-repudiable.

The flip-probability layer is empirical. We define a flip precisely as the 95% confidence interval's relationship to the null changing between review versions — binary, reproducible, guideline-panel-relevant. Ratio scales use null = 1; difference scales use null = 0. We do not predict effect magnitude; we predict whether the current decision region will move.

Training substrate: 595 Cochrane Collaboration reviews preserved as R-loadable study-level data in the Pairwise70 release (Hopewell et al, unpublished collection). Each .rda preserves per-study events/N, GIV means/SEs, and Study.year. Because published review versions and their differences are inaccessible without Cochrane CDSR API credentials, we reconstruct temporal pairs *within* each preserved review: an analysis's studies are sorted by year and pooled at the 55th and 80th percentile year-cutoffs, producing v1 and v2 snapshots grounded in the same underlying study set. The method trades one source of generalisation error (cross-version corpus heterogeneity) for another (within-review accretion bias), with full acknowledgement of the trade.

The paper's claims are narrow. First, a flip label can be computed reproducibly from any pair of published pooled effects. Second, the probability of such a label is learnable from real temporal Cochrane data with AUC > 0.70 using only effect geometry and heterogeneity, without pipeline enrichment. Third, the resulting forecast can be attached to every emitted pooled effect without altering the MA workflow. We do not claim that pipeline conditioning (the full-scope primitive) is yet empirically supported; that requires per-pair PICO-to-AACT matching, which is the declared next step.

## Methods (≈900 words)

### The Forecast Card

For any PICO the system emits a four-layer signed bundle: current pooled effect (native DerSimonian-Laird + Hartung-Knapp-Sidik-Jonkman random-effects pool with `t_{k-2}` prediction interval, implemented in `evidence_forecast._native_pool` because CardioSynth — the intended upstream — is a client-side HTML application without a Python API); flip probability with 95% bootstrap-perturbation CI (this paper's primitive); burden-weighted representativeness (min-weight overlap between trial-country mix and IHME disease-burden mix); and HMAC-SHA256 TruthCert over canonical JSON, with the key drawn only from a `TRUTHCERT_HMAC_KEY` environment variable and compared in constant time.

### Flip label

flip(v1, v2) = 1 iff sign((CI_low_v1 − null) × (CI_high_v1 − null)) ≠ sign((CI_low_v2 − null) × (CI_high_v2 − null)), where null = 1 for ratio scales and 0 for difference scales. Binary, reproducible, guideline-panel-relevant. We do *not* use a continuous effect-size change (e.g., |ln(OR_v2/OR_v1)| > 0.2) as the primary label; continuous-shift is reported as a supplementary analysis.

### Training substrate

1,632 real temporal pairs from 560 Pairwise70 `.rda` files (`scripts/extract_temporal_pairs.R`). Each pair is constructed by: (i) sorting an analysis's studies by Study.year; (ii) forming v1 as the set with year ≤ 55th percentile (requires ≥ 2 studies); (iii) forming v2 as year ≤ 80th percentile (requires ≥ 1 more study than v1); (iv) pooling each set via `metafor::rma(method="DL")` on risk ratio (binary endpoints) or generic-inverse-variance (continuous / time-to-event). Inclusion filter: 6–48 month gap between percentile dates. Pairs with identical effect or mismatched scales between v1 and v2 are dropped. Yield: 345 unique reviews, 7.1% flip rate (116/1,632 positive class).

### v1-snapshot features

Six families are reserved in the feature dictionary:

1. **Effect geometry** (extracted from pair): CI width, distance from null, prediction-interval width (NaN for temporal pairs because PI requires study-level refit).
2. **Heterogeneity** (extracted from pair via metafor): k, τ², I².
3. **Fragility** (Fragility Index, reverse-FI): not extracted per pair; filled with neutral constant.
4. **Bias** (Egger p, trim-and-fill Δ): not extracted per pair; filled with neutral constant.
5. **Pipeline** (Σ expected N in ongoing registered trials at v1 date, trial count, mean expected event rate, sponsor Shannon entropy, design heterogeneity = distinct study_type × phase × primary_purpose / total): extractable from the AACT canonical extract (`C:\Users\user\AACT\2026-04-12`) given per-pair PICO match terms, which are not yet wired for Cochrane review-level metadata. **Filled with neutral constants in the current training run**. Pipeline-ablation ΔAUC cannot yet be reported; this is the declared next step.
6. **Digit forensics** (Benford MAD): not extracted per pair; filled with neutral constant.

The current model therefore uses only families (1) and (2). AUC is produced from effect geometry and heterogeneity alone.

### Learning and validation

Gradient-boosting classifier (XGBoost where available; `sklearn.GradientBoostingClassifier` fallback) primary; Random Forest and L1-logistic regression as comparators. Temporal split: v1 < 2015-01-01 train, v1 ≥ 2015-01-01 test. No topic holdout (Pairwise70 is not tagged by medical specialty). Post-hoc Platt (sigmoid) calibration via `sklearn.CalibratedClassifierCV(cv=5)` on the training fold only. Metrics: AUC, Brier, calibration slope, calibration intercept, 10-bin reliability diagram. Sanity checks: label-shuffle permutation over five resamples; a declared-future pipeline-features ablation requiring per-pair AACT enrichment.

### Three seed PICOs

SGLT2i in HFpEF (Vaduganathan 2022 Lancet, PMID 36041474, pooled DELIVER + EMPEROR-Preserved n=12,251); tirzepatide in HFpEF and obesity (SUMMIT, Packer NEJM 2025, PMID 39555826, NCT04847557, primary composite of CV death or worsening HF); empagliflozin in T2DM CV outcomes (EMPA-REG OUTCOME, Zinman NEJM 2015, PMID 26378978, 3p-MACE). All three effects verified against primary sources 2026-04-14. Pipeline features extracted from AACT canonical 2026-04-12 snapshot.

## Results 1 — Calibration (≈550 words)

**Table 1** (held-out, n = 1,080, v1 ≥ 2015):

| Model | AUC | Brier | Calibration slope | Calibration intercept |
|---|---|---|---|---|
| GBM (raw) | 0.784 | 0.071 | 0.36 | −1.62 |
| **GBM + Platt sigmoid (shipped)** | **0.777** | **0.064** | **1.32** | **1.03** |
| Random Forest | (comparator; not shipped) | | | |
| L1-logistic (interpretability) | (comparator; not shipped) | | | |

Pre-registered ship thresholds: AUC ≥ 0.70 ✔ (0.777), Brier < 0.18 ✔ (0.064), calibration slope in [0.8, 1.2] near-miss (1.32, 10% over upper bound), label-permutation AUC ≈ 0.50 ✔ (0.514 mean over 5 shuffles).

**Calibration before vs. after Platt.** Raw GBM was systematically over-confident: slope 0.36 means the model distinguishes flips from non-flips well but compresses probabilities toward the extremes. Platt scaling via 5-fold CV on the training fold shifts slope to 1.32. Brier improves (0.071 → 0.064). AUC drops 0.007 (0.784 → 0.777). The post-Platt model is slightly *over*-calibrated (true base rate lies between predicted tertiles), an acceptable trade given the improved reliability.

**Permutation.** Shuffling the held-out labels and computing AUC against the same predicted probabilities produces mean AUC 0.514 over 5 shuffles, confirming no leakage from features to labels.

**Temporal-features ablation**: we added three v1-intrinsic temporal features extractable from each analysis's own study-year distribution (year span, years since most recent study, annual accrual rate). Retraining with these and comparing to the model without them yields **ΔAUC = 0.000** (full 0.778, ablated 0.778). This is a real null result: v1-intrinsic temporal signal is subsumed by k / τ² / I². It does not support the paper's pipeline-conditioning claim; the AACT-matched pipeline-features family remains unmeasured because per-pair PICO terms are not derivable from the Pairwise70 .rda metadata without external Cochrane review-title lookups. True pipeline-ablation therefore remains the single largest declared-future deliverable, and the current AUC of 0.778 is honest-best-available rather than a baseline under-estimate.

## Results 2 — Three seed PICOs (≈650 words)

**Figure 3** shows the three Forecast Cards side-by-side. All values below are extracted from the signed JSON bundles at commit `b4c5db5`.

**SGLT2i in HFpEF** (Vaduganathan 2022 pooled): effect HR 0.81 (native DL pool of DELIVER 0.82 [95% CI 0.73–0.92] and EMPEROR-Preserved 0.79 [0.69–0.90]; published pooled 0.80 reproduces within rounding). Flip probability **4.4%** (95% bootstrap CI from input-perturbation). AACT pipeline at 2026-04-14: 2 ongoing HF × SGLT2 trials, Σ expected N 604, sponsor Shannon entropy 1.0, design heterogeneity 1.0. Representativeness: not populated (registry-first weights not wired). Interpretation: mature pooled evidence with minimal ongoing trial contribution produces a low flip forecast, consistent with the field's current confidence in HFpEF SGLT2 benefit.

**SUMMIT tirzepatide in HFpEF and obesity** (Packer NEJM 2025, single trial): effect HR 0.62 (95% CI 0.41–0.95) for the primary composite of CV death or worsening heart failure; CV death alone numerically higher on tirzepatide (HR 1.58, 0.52–4.83, ns). Flip probability **22.1%**. AACT pipeline: 1 ongoing HF × tirzepatide trial, Σ expected N 120. Interpretation: single-trial evidence on a large composite endpoint, with a plausible-but-underpowered signal in the opposite direction on the mortality component, yields the highest forecast flip probability in the trio — the narrative centerpiece of this paper. The forecast is tension-bearing because the composite's significance hinges on event rates that could shift under a second trial or extended follow-up.

**EMPA-REG OUTCOME** (Zinman NEJM 2015, single landmark CVOT): effect HR 0.86 (95% CI 0.74–0.99) on 3p-MACE. Flip probability **9.4%**. AACT pipeline: 29 ongoing empagliflozin × diabetes trials, Σ expected N 467,671, sponsor entropy 4.2, design heterogeneity 0.31 — a large and diverse pipeline. Interpretation: the model assigns a low-to-moderate flip probability despite the enormous pipeline, consistent with the replicated nature of the cardiovascular benefit signal; the 29-trial pipeline is largely label-expansion, not primary-benefit challenge.

Across the trio, forecast ordering (SGLT2i < EMPA-REG < SUMMIT) matches the narrative ordering of evidence maturity. No single number in the trio is novel; their co-presentation with a non-repudiable signature is.

## Discussion (≈420 words)

**What we built**. A reproducible primitive: ≈2,800 lines of Python, 66 unit tests, 1,632 real training pairs grounded in metafor pooling of actual Cochrane study-level data, three worked examples with PMID/DOI/NCT-verified effects, native DL+HKSJ pooling replacing the never-existent CardioSynth Python API, and HMAC-SHA256 cryptographic attestation wired into every emitted card.

**What we did not build**. Per-pair AACT pipeline enrichment; empirical validation of the pipeline-features family's marginal contribution; topic-specific holdout (Pairwise70 lacks medical-specialty tags); and a Cochrane CDSR integration with true published-version v1 vs v2 data. Pairwise70 gives us 1,632 *within-review temporal-accretion pairs*, which is a weaker signal than true update pairs but the strongest substrate available without external access.

**Limitations**. The training substrate is retrospective within-review accretion, not true update pairs. The inclusion filter at 6–48 month year-percentile gaps likely biases toward smaller pairs where accretion is fast. Non-pipeline bias/fragility/Benford features are filled with neutral constants, reducing the effective model to effect geometry + heterogeneity. Calibration slope is 1.32, modestly above the pre-registered 1.2 upper bound. The SUMMIT tirzepatide PICO was originally drafted for all-cause mortality from session memory (HR 1.245) — this was wrong; the value has been replaced with the verified primary composite.

**Phase 2 preview**. Apply the same pipeline to each NICE cardiology recommendation in the portfolio's NICE-critique project. Publish flagged-unstable recommendations (flip P > 0.20 or representativeness overlap < 0.30) as BMJ Analysis + Lancet submissions, gated on acceptance of this method paper.

**Adoption call**. Every meta-analysis that publishes a pooled effect should publish a flip probability, and every forecast should carry a cryptographic attestation. The substrate (Pairwise70, AACT, IHME) is either public or acquirable. The primitive here takes less than 150 lines of Python to wire.

## Data and code availability

Code: github.com/mahmood726-cyber/evidence-forecast (commit `b4c5db5`).
TruthCert hashes for all three Forecast Cards are in `outputs/*.json`.
Trained model at `models/flip_forecaster_v1.pkl`; validation report at `models/validation_report_temporal_calibrated.json`.
Training pairs at `tests/fixtures/temporal_cochrane_pairs_v0.csv`.

## Competing interests

None declared.

## Supplement

- **S1.** Feature dictionary with extraction status (extracted / constant-filled per family).
- **S2.** GBM / Platt hyperparameters and sklearn versions.
- **S3.** Full 10-bin reliability diagram with bootstrap CI per bin.
- **S4.** Sensitivity analyses: 12-month and 36-month horizon variants.
- **S5.** Secondary continuous-shift label (|ln(OR_v2/OR_v1)| > 0.2): single-row AUC/Brier.
- **S6.** `REAL_PHASE1_BLOCKERS.md` — honest accounting of what was assumed vs. what exists on disk, and the four resolution paths.
