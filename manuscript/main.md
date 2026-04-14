# Evidence Forecasting: probabilistic prediction of meta-analytic conclusion stability

**Author**: Mahmood Ahmad, Tahir Heart Institute. ORCID 0009-0003-7781-4478.
**Target**: Nature Medicine (method paper).
**Status**: Draft — fill numerical results after real-data Task 17.

## Abstract

[~250 words — fill after real MetaAudit-trained model numbers land. Frame the primitive, the MetaAudit calibration, the cardiology holdout, and the three worked PICOs. Keep the flip-probability definition (CI-crosses-null) explicit. Lead with the claim that conclusion flip probability is learnable and not reducible to current MA internals.]

## Introduction (target 500 words)

Meta-analyses publish point estimates. Guideline panels decide binary recommendations. The translation step — will this conclusion survive the next wave of registered trials? — is performed informally, if at all. [Portfolio Manifesto background: evidence-intelligence stack, three layers Generate/Audit/Decide.] The opportunity: MetaAudit's longitudinal corpus supplies retrospective ground truth on which MA conclusions have flipped; AACT supplies the ongoing-trial substrate for a prospective forecast. Contribution: the first operational flip-forecast primitive with (a) a reproducible binary flip label, (b) a novel pipeline-features family as the forecasting lever, (c) cardiology held out for generalisation testing, and (d) cryptographic attestation attached to every emitted forecast.

## Methods (target 900 words)

### The Forecast Card

For any PICO the system emits a four-layer signed bundle: current effect (CardioSynth), flip probability with 95% bootstrap CI (this paper's primitive), burden-weighted representativeness (registry-first population layer), and HMAC-SHA256 TruthCert over the above.

### Flip label

flip(v1, v2) = 1 iff the 95% CI crosses the null in one MA version and not the other. Ratio scales (HR, OR, RR) use null = 1; difference scales (RD, MD, SMD) use null = 0. Binary, guideline-panel-relevant, reproducible.

### Feature set

Six families at the v1 snapshot: effect geometry (CI width, prediction-interval width, distance from null), heterogeneity (τ², I², Q-p, k), fragility (Fragility Index, reverse-FI from FragilityAtlas), bias (Egger p, trim-and-fill delta), pipeline (this paper's novelty: Σ expected N in ongoing registered trials at v1 date, trial count, mean expected event rate, sponsor Shannon entropy, design heterogeneity as distinct study_type × phase × primary_purpose / total ongoing trials), and digit forensics (Benford MAD).

### Training and validation

XGBoost (primary), Random Forest and L1-logistic regression (comparators). Temporal split: train on v1 ≤ 2022-12-31, test on v1 ≥ 2023-01-01. Cardiology MAs are entirely held out from training so the Phase-2 NICE application sits on genuinely unseen ground. Metrics: AUC, Brier, calibration slope and intercept, 10-bin reliability diagram. Sanity checks: label permutation must collapse AUC to ~0.5; a pipeline-features ablation quantifies novelty contribution as ΔAUC.

### Three seed PICOs

SGLT2i in HFpEF (Vaduganathan 2022 anchor), tirzepatide in HFpEF ACM (SUMMIT / Packer 2025 — narrative centerpiece), empagliflozin T2DM CV outcomes (EMPA-REG / Zinman 2015 benchmark).

## Results 1 — Calibration (target 700 words)

[Table 1: held-out metrics for GBM, RF, L1-LR — AUC, Brier, calibration slope, calibration intercept, n_test. Fill after real Task 17.]

[Figure 2: reliability diagram with 10 bins and bootstrap CI per bin. Expected shape: near-diagonal for calibrated model.]

[Pipeline-features ablation: ΔAUC between full feature set and the version with pipeline family dropped. Expected ≥ 0.02 to support the paper's novelty claim; if lower, reframe as "MA-internal features dominate; pipeline contributes marginal signal."]

[Permutation-shuffled AUC: report mean and 95% bootstrap interval over 100 shuffles. Expected ≈ 0.5; anything substantially off is a leakage alarm.]

## Results 2 — Trio (target 900 words)

[Figure 3: three Forecast Cards side-by-side. Real numbers after Task 17 replaces dev-mode stubs.]

[SGLT2i HFpEF: expected flip P low, representativeness high in Western ICU populations, pipeline modest.]

[SUMMIT tirzepatide ACM: expected flip P high — ACM HR non-significant at v1 but multiple large ongoing ACM-designed trials in the AACT pipeline push forecast upward. Narrative centerpiece — this is where the primitive earns its keep.]

[EMPA-REG: expected flip P very low — large, well-replicated, mature MA, pipeline signals stable.]

## Discussion (target 400 words)

Limitations: single training corpus (MetaAudit Cochrane reviews); cardiology held out means cardiology-specific generalisation is untested until Phase 2; pipeline-features matching is string-based in Phase 1 (MeSH-backed matching is Phase 2); flip label is binary and does not distinguish magnitude of effect-size change; 24-month horizon is arbitrary (12 and 36 month sensitivity analyses in supplement).

Phase 2 preview: apply the same pipeline to every NICE cardiology recommendation, publish flagged-unstable set, BMJ Analysis + Lancet submission.

Adoption call: every MA that publishes an effect should publish a flip probability, and every forecast should carry a cryptographic attestation. The primitive here is reproducible and the substrate (MetaAudit + AACT + IHME) is public.

## Data and code availability

Code: `github.com/mahmood726-cyber/evidence-forecast`. TruthCert hashes for all three Forecast Cards in supplementary material. Trained model artifact and validation report under `models/` in the repo.

## Competing interests

None.

## Supplement

- Feature dictionary.
- All hyperparameters for GBM, RF, L1-LR.
- Calibration curves.
- Sensitivity analyses: 12-month and 36-month horizons.
- Secondary continuous-shift label (|ln(OR_v2/OR_v1)| > 0.2): single-row AUC/Brier.
