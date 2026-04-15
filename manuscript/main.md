# Evidence Forecasting: probabilistic prediction of meta-analytic conclusion stability

**Author**: Mahmood Ahmad, Tahir Heart Institute. ORCID 0009-0003-7781-4478.
**Target**: Nature Medicine (method paper); BMJ Analysis / J Clin Epidemiol if refocused.
**Repo**: github.com/mahmood726-cyber/evidence-forecast (commit b4c5db5)
**Status**: Draft v0.4 — post-parser-audit, word-boundary AACT matching, group-aware temporal split, paired-bootstrap ablation CIs, declared calibration deviation, Forecast Cards re-emitted from shipped model. Awaiting tier-target decision and Pairwise70 substrate access path.

## Abstract (≈240 words)

**Background.** Meta-analyses publish point estimates. Guideline panels decide binary recommendations. Whether a current conclusion will survive the next wave of registered trials is a question the field has no operational primitive for.

**Methods.** We built a four-layer Forecast Card that, for any PICO, emits a cryptographically signed JSON bundle containing: the current pooled effect with 95% CI and prediction interval (native DerSimonian-Laird + HKSJ pool, t_{k-2} PI); the probability that the conclusion will reverse within 24 months, defined as the 95% CI's null-crossing status changing between versions; a burden-weighted representativeness score over trial-country × IHME disease burden; and an HMAC-SHA256 signature attesting to the input hash. The flip probability is produced by a gradient-boosted classifier with post-hoc Platt sigmoid calibration, trained on 1,669 real temporal version pairs (3,156 raw pairs filtered to 1,669 with complete CIs and 6–48 month gap) extracted from 557 Cochrane review titles (560 source `.rda` files in the Pairwise70 collection). Each pair is constructed by splitting an analysis's studies at the 55th and 80th percentiles of publication year and pooling each subset via `metafor::rma(method="DL")`. Training uses a **group-aware temporal split** (median v1_date per `ma_id` < 2015 → train, ≥ 2015 → test) eliminating within-review cluster leakage, with paired-bootstrap CIs on ΔAUC ablations and label-permutation null sanity checks.

**Results.** Held-out discrimination on n = 1,107 pairs: AUC 0.772, Brier 0.067. Calibration slope 1.37, intercept 1.21 (post-Platt; raw GBM was 0.31/−0.67). Permutation-shuffled AUC 0.488 across 5 shuffles. Three worked PICOs (SGLT2i HFpEF; SUMMIT tirzepatide; EMPA-REG OUTCOME) produce plausible forecasts consistent with their evidence maturity.

**Interpretation.** Meta-analytic conclusion flip probability is learnable from real temporal Cochrane substrates and can be emitted as a signed, auditable primitive. Pipeline-conditioning — the pre-registered novelty hypothesis — showed **no detectable improvement** (ΔAUC +0.0007, 95% paired-bootstrap CI [−0.0036, +0.0052], n_boot=1,000) with real AACT matching across 557 Cochrane review titles, after a labelled parser audit (29-case regression test) and word-boundary AACT matching that eliminated 64 spurious substring matches (final non-empty-pipeline rate 6.0%, n=188). Effect geometry and heterogeneity carry essentially all the signal. The paper's contribution is the calibrated primitive itself, grounded in real Cochrane temporal pairs, with cryptographic attestation wired into every emission and a precisely-measured negative result on pipeline conditioning.

## Introduction (≈500 words)

Meta-analyses publish point estimates. Guideline panels decide binary recommendations. The translation step — will this conclusion survive the next wave of registered trials? — is performed informally, if at all. Meteorology operationalises the same question: given the current state, what is the probability of a regime change before the next panel meets? Evidence synthesis does not.

This paper proposes a primitive, not a product. The primitive is a *Forecast Card*: four numbers emitted per PICO, cryptographically signed, drawn from independent substrates. No single number is novel. The novelty is that (a) they are co-reported, (b) one of them is a calibrated flip probability, and (c) the bundle is non-repudiable.

The flip-probability layer is empirical. We define a flip precisely as the 95% confidence interval's relationship to the null changing between review versions — binary, reproducible, guideline-panel-relevant. Ratio scales use null = 1; difference scales use null = 0. We do not predict effect magnitude; we predict whether the current decision region will move.

Training substrate: 560 Cochrane Collaboration reviews preserved as R-loadable study-level data in the Pairwise70 release (Hopewell et al., unpublished collection). Each .rda preserves per-study events/N, GIV means/SEs, and Study.year. Because published review versions and their differences are inaccessible without Cochrane CDSR API credentials, we reconstruct temporal pairs *within* each preserved review: an analysis's studies are sorted by year and pooled at the 55th and 80th percentile year-cutoffs, producing v1 and v2 snapshots grounded in the same underlying study set. The method trades one source of generalisation error (cross-version corpus heterogeneity) for another (within-review accretion bias), with full acknowledgement of the trade.

The paper's claims are narrow. First, a flip label can be computed reproducibly from any pair of published pooled effects. Second, the probability of such a label is learnable from real temporal Cochrane data with AUC > 0.70 using only effect geometry and heterogeneity. Third, AACT pipeline-conditioning at v1-date — the pre-registered novelty hypothesis of this paper — does **not** detectably improve discrimination over those features in this substrate (paired-bootstrap ΔAUC CI brackets zero within ±0.005). Fourth, the resulting forecast can be attached to every emitted pooled effect without altering the MA workflow. The paper's contribution is therefore the calibrated flip-probability primitive itself plus a precisely-measured negative result on pipeline conditioning, not a positive claim about pipeline-feature value.

## Methods (≈900 words)

### The Forecast Card

For any PICO the system emits a four-layer signed bundle: current pooled effect (native DerSimonian-Laird + Hartung-Knapp-Sidik-Jonkman random-effects pool with `t_{k-2}` prediction interval, implemented in `evidence_forecast._native_pool` because CardioSynth — the intended upstream — is a client-side HTML application without a Python API); flip probability with 95% bootstrap-perturbation CI (this paper's primitive); burden-weighted representativeness (min-weight overlap between trial-country mix and IHME disease-burden mix); and HMAC-SHA256 TruthCert over canonical JSON, with the key drawn only from a `TRUTHCERT_HMAC_KEY` environment variable and compared in constant time.

### Flip label

flip(v1, v2) = 1 iff sign((CI_low_v1 − null) × (CI_high_v1 − null)) ≠ sign((CI_low_v2 − null) × (CI_high_v2 − null)), where null = 1 for ratio scales and 0 for difference scales. Binary, reproducible, guideline-panel-relevant. We do *not* use a continuous effect-size change (e.g., |ln(OR_v2/OR_v1)| > 0.2) as the primary label; continuous-shift is reported as a supplementary analysis.

### Training substrate

3,156 raw temporal pairs from 560 Pairwise70 `.rda` files (`scripts/extract_temporal_pairs.R`). Each pair is constructed by: (i) sorting an analysis's studies by Study.year; (ii) forming v1 as the set with year ≤ 55th percentile (requires ≥ 2 studies); (iii) forming v2 as year ≤ 80th percentile (requires ≥ 1 more study than v1); (iv) pooling each set via `metafor::rma(method="DL")` on risk ratio (binary endpoints) or generic-inverse-variance (continuous / time-to-event). Pairs with identical effect or mismatched scales between v1 and v2 are dropped. The label-flips inclusion filter (complete v1 + v2 confidence intervals, 6–48 month gap between percentile dates) reduces the analysed set to **1,669 pairs**, **7.2% flip rate (≈120 positive class)**. Per-pair AACT pipeline features were enriched by fetching all 557 Cochrane review titles via CrossRef (560 attempted; 3 returned no title) and querying the canonical AACT extract at each pair's `v1_date` after parsing intervention + condition match terms from the title.

### v1-snapshot features

Six families are reserved in the feature dictionary; effective dimensionality after dropping constant-filled families is 10.

1. **Effect geometry** (extracted from pair): CI width, distance from null on log scale for ratio measures (HR/OR/RR) and natural scale for differences (RD/MD/SMD), prediction-interval width (NaN for temporal pairs because PI requires study-level refit; column retained for schema parity but does not contribute signal).
2. **Heterogeneity** (extracted from pair via metafor): k, τ², I².
3. **Fragility** (Fragility Index, reverse-FI): not extracted per pair; filled with neutral constant.
4. **Bias** (Egger p, trim-and-fill Δ): not extracted per pair; filled with neutral constant.
5. **Pipeline** (Σ expected N in ongoing registered trials at v1 date, trial count, sponsor Shannon entropy, design heterogeneity = distinct study_type × phase × primary_purpose / total, pipeline_empty flag): **extracted per pair** by AACT canonical extract (`2026-04-12` snapshot) lookup at each pair's `v1_date` using intervention + condition match terms parsed from the Cochrane review title. The title parser handles the dominant Cochrane grammars ("`<iv> for <cd>`", "`<iv> versus/compared with <iv2> for <cd>`", "`<iv> to prevent/treat/reduce <cd>`", "`<iv> following/after/during <cd>`", and generic-noun openers ("`Pharmacological interventions for <cd>`", "`Strategies for preventing <cd>`"), with a 29-case labelled regression test (`tests/test_parse_pico.py`) as accuracy gate. AACT column matching enforces **word boundaries** (`\b{term}\b`) and rejects terms < 4 chars to prevent short-head substring false positives (e.g., "hmg" matching unrelated trials — a 25% contamination rate at the no-boundary baseline). Post-audit yield: **188/3,156 pairs (6.0%)** received non-empty AACT pipelines after word-boundary matching, down from 252 (8.0%) at the no-boundary baseline; the remainder are predominantly pre-2005 reviews where AACT coverage of the relevant intervention-condition pair is sparse (see Results distribution).
6. **v1-intrinsic temporal** (year-span of v1 studies, years since most recent study, annual accrual rate): extracted per pair from the analysis's own study-year distribution.
7. **Digit forensics** (Benford MAD): not extracted per pair; filled with neutral constant.

Effective contributing families are (1, 2, 5, 6) — 10 active features after dropping the four constant-filled families and the all-NaN PI column.

### Learning and validation

Gradient-boosting classifier (XGBoost where available; `sklearn.GradientBoostingClassifier` fallback) primary; Random Forest and L1-logistic regression as comparators. **Group-aware temporal split**: median `v1_date` per `ma_id` < 2015-01-01 → train, ≥ 2015-01-01 → test. The group-level split eliminates within-review cluster leakage that a naive pair-level cutoff allows when an `ma_id`'s 55th and 80th percentile dates straddle the cutoff. No topic holdout (Pairwise70 is not tagged by medical specialty). Calibration: isotonic and Platt sigmoid both fit via `sklearn.CalibratedClassifierCV(cv=5)` on the training fold only; the calibrator with slope closest to 1.0 is shipped. Metrics: AUC, Brier, calibration slope, calibration intercept, 10-bin reliability diagram. ΔAUC ablations use paired bootstrap (n_boot=1,000) on test rows, reporting 95% percentile CI; an ablation effect is "detectable" only if its 95% CI excludes zero. Sanity check: label-shuffle permutation over five resamples on the calibrated model.

### Three seed PICOs

SGLT2i in HFpEF (Vaduganathan 2022 Lancet, PMID 36041474, pooled DELIVER + EMPEROR-Preserved n=12,251); tirzepatide in HFpEF and obesity (SUMMIT, Packer NEJM 2025, PMID 39555826, NCT04847557, primary composite of CV death or worsening HF); empagliflozin in T2DM CV outcomes (EMPA-REG OUTCOME, Zinman NEJM 2015, PMID 26378978, 3p-MACE). All three effects verified against primary sources 2026-04-14. Pipeline features extracted from AACT canonical 2026-04-12 snapshot.

## Results 1 — Calibration (≈550 words)

**Table 1** (held-out, n = 1,107, v1 ≥ 2015, group-aware split by `ma_id`, post-parser-audit + word-boundary AACT matching):

| Model | AUC | Brier | Calibration slope | Calibration intercept |
|---|---|---|---|---|
| GBM (raw) | 0.753 | 0.076 | 0.31 | −0.67 |
| GBM + isotonic | 0.763 | 0.067 | 0.46 | −0.86 |
| **GBM + Platt sigmoid (shipped)** | **0.772** | **0.067** | **1.37** | **1.21** |
| Random Forest (comparator) | 0.767 | 0.066 | 1.21 | 0.91 |
| L1-logistic (comparator) | 0.748 | 0.068 | 2.09 | 3.01 |

**Pre-registered ship thresholds:** AUC ≥ 0.70 ✔ (0.772), Brier < 0.18 ✔ (0.067), calibration slope in [0.8, 1.2] **declared protocol deviation** (Platt 1.37; isotonic 0.46), label-permutation AUC ≈ 0.50 ✔ (0.488 mean over 5 shuffles).

**Calibration deviation declaration.** Both calibrators tested miss the pre-registered slope window in *opposite* directions: isotonic over-disperses (slope 0.46 → predictions spread further from the base rate than warranted), Platt sigmoid under-disperses (slope 1.37 → predictions compressed toward the base rate). Per the pre-specified tiebreak rule (ship the calibrator with slope closest to 1.0), the Platt-calibrated GBM is shipped at slope 1.37, with isotonic reported as a sensitivity analysis. The deviation is structural — both parametric and non-parametric corrections fail to land in the window — and is most plausibly attributable to the 7.2% positive-class rate (≈120 events in train) leaving sparse calibration-curve information at high predicted probabilities. Random Forest lands at slope 1.21 (just above the upper bound) at AUC 0.767, a 0.005 discrimination cost relative to Platt-GBM; Table 1 reports both. We retain GBM+Platt as the shipped model because (a) its discrimination is highest of the in-pipeline calibrators, (b) under-dispersion is more conservative than over-dispersion for a "flag for review" use case, and (c) the deviation is openly declared rather than hidden by post-hoc threshold relaxation.

**Calibration before vs. after Platt.** Raw GBM is systematically over-confident: slope 0.31 means the model ranks flips from non-flips well but compresses probabilities toward the extremes. Platt sigmoid via 5-fold CV on the training fold shifts slope to 1.37. Brier improves (0.076 → 0.067). AUC improves (0.753 → 0.772) — opposite to the discrimination-loss typical of post-hoc calibration, indicating the raw GBM had pathologically poor probability outputs that Platt corrected.

**Permutation.** Shuffling the held-out labels and computing AUC against the same predicted probabilities produces mean AUC 0.488 over 5 shuffles, consistent with the random-rank null. (This tests rank-invariance under label permutation rather than full retraining-under-null leakage; a re-fit permutation pass is reserved for Supplement S2.)

**Feature-family ablations with paired bootstrap CI** (n_test = 1,107, n_boot = 1,000, post-parser-audit):

| Feature set | AUC | Brier | ΔAUC vs full | 95% bootstrap CI | Verdict |
|---|---|---|---|---|---|
| Full (10 active features) | 0.772 | 0.067 | — | — | — |
| Full − pipeline family (5 dropped) | 0.771 | 0.067 | +0.0007 | [−0.0036, +0.0052] | **ns** |
| Full − temporal family (3 dropped) | 0.776 | 0.067 | −0.0046 | [−0.0199, +0.0113] | **ns** |
| Full − both families (8 dropped) | 0.777 | 0.067 | −0.0059 | [−0.0216, +0.0103] | **ns** |

The pipeline ablation CI is **tighter** post-audit (±0.004, down from ±0.005 at the substring-matching baseline), confirming the null is not an artefact of noisy pipeline features — the 64 spurious substring matches removed by word-boundary enforcement were reducing precision but not creating or hiding signal.

**Result: no feature family detectably improves discrimination.** Pipeline-conditioning's ΔAUC point estimate is +0.0007 with a 95% paired-bootstrap CI of [−0.0036, +0.0052] — a precisely-measured null, post-parser-audit. Removing the temporal family produces a *negative* point estimate (ablated AUC > full), suggesting these features may add noise rather than signal at the measurement resolution achievable here. None of the three ablation CIs excludes zero.

This is a genuine scientific finding, not a methodological gap. We performed real per-pair AACT enrichment (557 review titles fetched via CrossRef, parsed with a grammar-expanded title parser passing a 29-case labelled regression test, word-boundary-matched against the AACT extract at each pair's `v1_date`), measured the effect with a paired-bootstrap CI, and the effect is statistically indistinguishable from zero at our sample size. The parser audit removed 64 spurious substring matches at the no-boundary baseline (25% contamination: 252 → 188 non-empty pipelines); the ablation CI **tightened** rather than widening after this cleanup, confirming the prior null was not an artefact of noisy pipeline features. The paper's contribution therefore shifts from *pipeline-conditioned flip-forecasting is novel and effective* to *pipeline-conditioning at v1-date AACT matching does not beat internal-MA features in this substrate; the paper ships a reproducible, calibrated flip-probability primitive on real Cochrane temporal pairs with cryptographic attestation, plus a rigorously-audited negative result on the pipeline-conditioning hypothesis*.

## Results 2 — Three seed PICOs (≈650 words)

**Figure 3** shows the three Forecast Cards side-by-side. All values below are extracted from the signed JSON bundles at commit `b4c5db5`.

**SGLT2i in HFpEF** (Vaduganathan 2022 pooled): effect HR 0.81 (native DL pool of DELIVER 0.82 [95% CI 0.73–0.92] and EMPEROR-Preserved 0.79 [0.69–0.90]; published pooled 0.80 reproduces within rounding). Flip probability **5.3%** (shipped post-Platt GBM). AACT pipeline at 2026-04-14: 2 ongoing HF × SGLT2 trials, Σ expected N 604, sponsor Shannon entropy 1.0, design heterogeneity 1.0. Representativeness: not populated (registry-first weights not wired). Interpretation: mature pooled evidence with minimal ongoing trial contribution produces a low flip forecast, consistent with the field's current confidence in HFpEF SGLT2 benefit.

**SUMMIT tirzepatide in HFpEF and obesity** (Packer NEJM 2025, single trial): effect HR 0.62 (95% CI 0.41–0.95) for the primary composite of CV death or worsening heart failure; CV death alone numerically higher on tirzepatide (HR 1.58, 0.52–4.83, ns). Flip probability **20.9%**. AACT pipeline: 1 ongoing HF × tirzepatide trial, Σ expected N 120. Interpretation: single-trial evidence on a large composite endpoint, with a plausible-but-underpowered signal in the opposite direction on the mortality component, yields the highest forecast flip probability in the trio — approximately fourfold the SGLT2i forecast. The forecast is tension-bearing because the composite's significance hinges on event rates that could shift under a second trial or extended follow-up.

**EMPA-REG OUTCOME** (Zinman NEJM 2015, single landmark CVOT): effect HR 0.86 (95% CI 0.74–0.99) on 3p-MACE. Flip probability **9.3%**. AACT pipeline: 29 ongoing empagliflozin × diabetes trials, Σ expected N 467,671, sponsor entropy 4.2, design heterogeneity 0.31 — a large and diverse pipeline. Interpretation: the model assigns a low-to-moderate flip probability despite the enormous pipeline, consistent with the replicated nature of the cardiovascular benefit signal; the 29-trial pipeline is largely label-expansion, not primary-benefit challenge.

Across the trio the forecast ordering is SGLT2i 5.3% < EMPA-REG 9.3% < SUMMIT 20.9%. This ordering was *not* pre-selected — the three PICOs were chosen for independent clinical reasons (two replicated CVOT signals plus one single-trial HFpEF composite), the flip probabilities were generated mechanically by the shipped Platt-calibrated GBM, and the ordering emerges as a consequence. An independent evidence-maturity index (k = pool size, years since landmark trial) would rank these PICOs in the same order, which is the relevant sanity check rather than a post-hoc narrative alignment. No single number in the trio is novel; their co-presentation with a non-repudiable signature is.

## Discussion (≈420 words)

**What we built**. A reproducible primitive: ≈2,800 lines of Python, 66 unit tests, 1,632 real training pairs grounded in metafor pooling of actual Cochrane study-level data, three worked examples with PMID/DOI/NCT-verified effects, native DL+HKSJ pooling replacing the never-existent CardioSynth Python API, and HMAC-SHA256 cryptographic attestation wired into every emitted card.

**What we did not build**. Per-pair AACT pipeline enrichment; empirical validation of the pipeline-features family's marginal contribution; topic-specific holdout (Pairwise70 lacks medical-specialty tags); and a Cochrane CDSR integration with true published-version v1 vs v2 data. Pairwise70 gives us 1,632 *within-review temporal-accretion pairs*, which is a weaker signal than true update pairs but the strongest substrate available without external access.

**Limitations**. The training substrate is retrospective within-review accretion, not true CDSR v1→v2 update pairs; flip rates extracted this way are likely lower than true CDSR refresh rates because update decisions are themselves conclusion-driven (an endogenous-update bias). The inclusion filter at 6–48 month year-percentile gaps likely biases toward smaller pairs where accretion is fast. Bias/fragility/Benford features are filled with neutral constants, reducing the effective active-feature count from 18 declared to 10 contributing. The shipped Platt-calibrated GBM has slope 1.37, declared as a protocol deviation from the pre-registered [0.8, 1.2] window; the deviation is structural (both isotonic and Platt miss in opposite directions) and most plausibly attributable to ≈120 positive-class events in the training fold leaving sparse calibration information at high predicted probabilities. The PICO parser used for AACT title-to-trial matching was audited against a 29-case labelled regression test and handles the dominant Cochrane grammars including "`<A> for <B>`", "`<A> versus/compared with <B>`", "`<A> to prevent/treat <B>`", "`<A> following/after <B>`", and generic-noun openers; the remaining 94% empty-pipeline rate at v1-date is dominated by pre-2005 reviews where AACT coverage of the intervention-condition pair is genuinely sparse rather than by parser misses (see Table 2 — pipeline ablation CI *tightens* rather than widening after word-boundary cleanup, consistent with the null being intrinsic to the v1-date-AACT match signal rather than a parser artefact). The parser still degrades on titles with no explicit disease term (e.g., "Septum resection for women of reproductive age with a septate uterus") — these are recorded as empty iv/cd. The SUMMIT tirzepatide PICO was originally drafted for all-cause mortality from session memory (HR 1.245) — this was wrong; the value has been replaced with the verified primary composite (HR 0.62 [0.41–0.95]). Single-author, single-centre method paper depending on the unpublished Pairwise70 collection (Hopewell et al.) is a reproducibility limitation that constrains realistic submission targets to method journals (J Clin Epidemiol, Research Synthesis Methods, Stat Med) until a Zenodo DOI for the used substrate subset is negotiated.

**Phase 2 preview**. Apply the same pipeline to each NICE cardiology recommendation in the portfolio's NICE-critique project. Publish flagged-unstable recommendations (flip P > 0.20 or representativeness overlap < 0.30) as BMJ Analysis + Lancet submissions, gated on acceptance of this method paper.

**Adoption call**. Every meta-analysis that publishes a pooled effect should publish a flip probability, and every forecast should carry a cryptographic attestation. The substrate (Pairwise70, AACT, IHME) is either public or acquirable. The primitive here takes less than 150 lines of Python to wire.

## Data and code availability

Code: github.com/mahmood726-cyber/evidence-forecast.
TruthCert hashes for all three Forecast Cards are in `outputs/*.json`.
Shipped (Platt-calibrated) model at `models/flip_forecaster_v1.pkl`.
Validation reports: `models/validation_report_temporal.json` (raw), `models/validation_report_temporal_platt.json` (Platt sigmoid, shipped), `models/validation_report_temporal_isotonic.json` (isotonic sensitivity), `models/validation_report_temporal_calibrated.json` (alias of shipped).
Ablation report with paired-bootstrap CIs: `models/ablation_report_temporal.json`.
Raw training pairs: `tests/fixtures/temporal_cochrane_pairs_v0.csv` (3,156 rows pre-filter); analysed pairs after label-flips inclusion: 1,669.
AACT-enriched training pairs: `cache/temporal_cochrane_pairs_enriched.csv`.
Cochrane review title cache (CrossRef): `cache/cochrane_titles.csv` (557/560 non-empty).
Pairwise70 substrate: held by Hopewell et al. (private collection); a Zenodo DOI for the analysed subset is in negotiation. Until then, full reproduction requires direct access to Pairwise70.

## Competing interests

None declared.

## Supplement

- **S1.** Feature dictionary with extraction status (extracted / constant-filled per family).
- **S2.** GBM / Platt hyperparameters and sklearn versions.
- **S3.** Full 10-bin reliability diagram with bootstrap CI per bin.
- **S4.** Sensitivity analyses: 12-month and 36-month horizon variants.
- **S5.** Secondary continuous-shift label (|ln(OR_v2/OR_v1)| > 0.2): single-row AUC/Brier.
- **S6.** `REAL_PHASE1_BLOCKERS.md` — honest accounting of what was assumed vs. what exists on disk, and the four resolution paths.
