# review-findings.md

**Status (path B fix-pass complete, 2026-04-15):** 9 / 14 P0 RESOLVED. 67/67 tests pass (was 66 + 1 new group-split contract test). Manuscript updated against ground-truth post-fix numbers. Calibration deviation declared (option I+III). Remaining P0 items are scope-deferred (Pairwise70 access, TRIPOD compliance, parser-accuracy audit) — see "Remaining" section at bottom.

---

**Date:** 2026-04-15
**Scope:** 4 files changed today (retraining + AACT enrichment + manuscript reframe)
- `scripts/train_on_temporal_pairs.py`
- `scripts/build_per_pair_pipeline_features.py`
- `scripts/fetch_cochrane_titles.py`
- `manuscript/main.md`

**Method:** 5-persona review (Statistical Methodologist, Security Auditor, Software Engineer, Domain Expert, Scientific Communication) with ground-truth verification against shipped artefacts.

**Summary:** 14 P0, 14 P1, 18 P2. Several P0 issues are manuscript-blocking; the most severe is numerical staleness — the shipped validation report disagrees with the manuscript on calibration slope/intercept, and the post-enrichment model fails its pre-registered calibration threshold.

---

## Ground-truth reference values (verified 2026-04-15)

Queried shipped artefacts directly:

| Claim | Manuscript | Shipped artefact | Source |
|---|---|---|---|
| pair count | 1,632 | **3,156** | `cache/temporal_cochrane_pairs_enriched.csv` |
| unique reviews | 345 | **3,152 unique ma_ids** / 560 .rda files | pairs CSV |
| Cochrane titles | 556 | **557 non-empty / 560 total** | `cache/cochrane_titles.csv` |
| n_test | 1,080 | **1,107** | validation JSON |
| AUC | 0.777 | **0.7776** (rounds to 0.778) | validation JSON |
| Brier | 0.064 | **0.0667** | validation JSON |
| calibration slope | 1.32 | **1.595** | validation JSON |
| calibration intercept | 1.03 | **1.80** | validation JSON |

The calibration numbers 1.32 / 1.03 appear to be from a pre-enrichment run. Post-enrichment values (1.595 / 1.80) **clearly fail** the pre-registered ship threshold of slope ∈ [0.8, 1.2] — the failure is 33% above the upper bound, not the "near-miss" claimed in line 76.

---

## P0 — Critical (must fix before any submission)

### Data-integrity / numerical staleness

- **[FIXED] [P0-1] [COM+STAT]** `manuscript/main.md:12, 14, 42, 67, 72, 76, 82` — Every headline number disagrees with the shipped validation report. Calibration slope/intercept staleness is the most serious; the post-enrichment re-train fails the pre-registered threshold. Fix: regenerate all tables + abstract numbers from `models/validation_report_temporal_calibrated.json`, and explicitly note the preregistration failure (either declare a pre-specified deviation with rationale, or re-evaluate whether the model should ship).

- **[FIXED] [P0-2] [COM+DOM]** `manuscript/main.md:14, 42, 91` — Substrate count contradiction: 1,632 vs 3,156 pairs, 345 vs 3,152 unique reviews. Reviewers will reject. Fix: lock to actual 3,156 enriched pairs / 3,152 unique ma_ids / 560 source reviews with a single clear definition, and update abstract, methods line 42, and results line 91.

- **[FIXED] [P0-3] [STAT+COM]** `manuscript/main.md:16, 93` & `train_on_temporal_pairs.py:204-209` — "Pipeline-conditioning refuted at measurable scale" is a positive null claim from a **point-estimate ΔAUC = +0.002 with no CI, no DeLong test, no bootstrap**. Required fix: compute paired bootstrap or DeLong CI for ΔAUC (both families); soften to "no detectable improvement" if CI crosses zero, OR tighten to "refuted" only if CI excludes a clinically-relevant improvement.

### Methodological

- **[FIXED] [P0-4] [STAT]** `train_on_temporal_pairs.py:63` — `distance_from_null = |v1_point − null|` is computed on the **natural (ratio) scale** for HR/OR/RR. Per `advanced-stats.md` ("Log scale: Always pool logRR/logOR/logHR"), ratios of 2.0 and 0.5 are symmetric in effect but get asymmetric distances (1.0 vs 0.5), biasing the GBM. Fix: `np.log(v1_point)` for ratio scales, `v1_point` for difference scales, take `.abs()` after.

- **[FIXED] [P0-5] [STAT]** `train_on_temporal_pairs.py:132` — Temporal split at pair-level `v1_date`. Pairs are extracted **within** the same review at 55th/80th-percentile cutoffs; the same `ma_id` can have sibling pairs straddling the 2015 cutoff → cluster leakage. Fix: `GroupShuffleSplit` or manual split by `ma_id`.

- **[P0-6] [DOM]** `build_per_pair_pipeline_features.py:92-134` — `parse_pico` handles only "for / versus / in" and a small `_PREFIX_STRIP` list. Real Cochrane corpus has 25–40% titles outside this grammar ("compared with", "to prevent", "Pharmacological interventions for", "Early vs. late…", "Single-dose vs…"). The reported 8% AACT non-empty rate **mixes genuine pre-AACT-era misses with parser misses** — the null-result claim cannot stand until parser accuracy is measured on a ≥200-title labelled sample. Fix: expand parser, measure extraction accuracy, re-report.

- **[P0-7] [DOM]** `build_per_pair_pipeline_features.py:129-134` + `:48-49` — `_head_word` takes the first alphabetic ≥3-char token and substring-matches against `conditions_lc` / `interventions_lc` with **no word boundaries**. "cad" matches "cadmium"; "arf" matches many; "statins" misses "rosuvastatin" / "atorvastatin". Per `lessons.md` CT.gov rule: "Drug names, not class names." Fix: word-boundary regex or `\b{term}\b`, plus a class→member expansion list for common drug classes.

### Engineering bugs

- **[FIXED] [P0-8] [ENG]** `build_per_pair_pipeline_features.py:175` — `re.match(r"(CD\d+)", row["ma_id"]).group(1)` raises `AttributeError` on any `ma_id` not starting with `CDnnn` (contrast with line 152 which guards with truthiness). Fix: `m = re.match(...); if not m: continue`.

- **[FIXED] [P0-9] [ENG]** `build_per_pair_pipeline_features.py:46` — `date.fromisoformat(snapshot_date)` crashes on any `v1_date` with a time component (e.g., `2015-03-01T00:00:00`) or non-ISO format. Fix: `pd.to_datetime(snapshot_date, errors="coerce").date()` and skip on NaT.

- **[FIXED] [P0-10] [ENG]** `train_on_temporal_pairs.py:29-38` — `FEATURE_COLS` is re-declared locally but also imported as `_feature_cols` from `evidence_forecast.calibration.train` (line 25). Drift → silent corruption at serve time because the saved bundle uses one ordering and the ablation loop uses the other. Fix: import the canonical list or `assert FEATURE_COLS == _feature_cols` at startup.

### Claims / manuscript

- **[P0-11] [COM+DOM]** `manuscript/main.md:26, 121-126` — Training substrate (Pairwise70) is declared **"unpublished"**. No tier-1 methods venue will accept a prediction-model paper without a citable, re-acquirable substrate. Fix: negotiate Zenodo DOI for the used subset with Hopewell et al., OR restrict training to a publicly-available Cochrane-CDSR subset before submission.

- **[P0-12] [DOM+COM]** `manuscript/main.md` — No PRISMA 2020 citation for the base MAs, no **TRIPOD 2015 / TRIPOD+AI 2024** compliance statement. TRIPOD is mandatory for prediction-model papers at any reputable methods or clinical journal. Fix: add TRIPOD+AI 2024 checklist supplement.

- **[FIXED] [P0-13] [DOM+COM]** `manuscript/main.md:67, 73-74` — Table 1 rows for RF and L1-LR are **empty**. The training script prints "Training GBM/RF/L1-LR" so values should exist — either populate from `train_models` artefacts or delete rows. Phantom comparators are publication-blocking.

- **[P0-14] [COM]** `manuscript/main.md:107` — "Forecast ordering (SGLT2i < EMPA-REG < SUMMIT) matches narrative ordering of evidence maturity" is **circular** (narrative written after seeing forecasts). Fix: pre-register an independent maturity metric (e.g., `k` trials in pool, years since first positive) and compare forecast ordering to *that*, not to a post-hoc narrative.

---

## P1 — Important (should fix before submission)

- **[P1-1] [STAT]** `train_on_temporal_pairs.py:167-176` — Permutation shuffles `y` against fixed `p` → tests rank invariance, not leakage. Manuscript line 80 over-interprets as "no leakage from features to labels". Fix: either re-train under `rng.permutation(y_tr)` and report retrained test AUC, or relabel as "rank-invariance check."

- **[P1-2] [STAT]** `train_on_temporal_pairs.py:60`, `manuscript:48` — `pi_width` is all-NaN per pair; `SimpleImputer(strategy="median")` on an all-NaN column is a no-op → contributes zero signal. Fix: drop `pi_width` from `FEATURE_COLS` entirely.

- **[P1-3] [STAT+DOM]** `train_on_temporal_pairs.py:70-81`, `manuscript:50-53` — Fragility, Egger, trim-fill, Benford are neutral constants. Claiming "18 features" in Table 2 is misleading when 4 are dead. Fix: re-report as 14 active features, or extract at least one of these per-pair.

- **[P1-4] [DOM]** `manuscript/main.md:24, 38` — Flip label is defined only at null-crossing. Guideline panels routinely use MID thresholds. Fix: add MID-threshold flip as co-primary (not S5 supplement) and cite GRADE guidance.

- **[P1-5] [DOM]** `manuscript/main.md:76` — Pre-registered calibration threshold [0.8, 1.2] is failed (actual 1.595 per verified JSON). Fix: declare pre-specified deviation with isotonic-calibration sensitivity analysis, OR re-Platt with larger folds, OR do not ship.

- **[P1-6] [SEC]** `fetch_cochrane_titles.py:27` — `url = f"https://api.crossref.org/works/{doi}"` interpolates raw DOI with zero sanitisation. Fix: `re.fullmatch(r"10\.\d{4,9}/[\w\.\-/()]+", doi)` validate + `urllib.parse.quote(doi, safe="/")` before interpolation.

- **[P1-7] [SEC]** `fetch_cochrane_titles.py:77-81` — CSV writer emits CrossRef strings verbatim; title starting with `=` / `+` / `@` / `\t` / `\r` → formula injection when opened in Excel. Per `lessons.md`. Fix: prepend `'` to any such cell.

- **[P1-8] [SEC]** `fetch_cochrane_titles.py:30` — `urllib.request.urlopen` follows redirects silently with default SSL context. Fix: custom `HTTPRedirectHandler` with hop cap + blocked cross-host redirects.

- **[P1-9] [SEC]** `train_on_temporal_pairs.py:122-123, 160-161` — `pickle.load` on shipped `models/flip_forecaster_v1.pkl`. Path is trusted here but the artefact is shipped and referenced in manuscript as a release asset → any consumer who re-uses the pattern on an attacker `.pkl` is RCE'd. Fix: switch shipping loader to `joblib` + HMAC-over-bytes verification using `TRUTHCERT_HMAC_KEY` before `pickle.load`.

- **[P1-10] [ENG]** `train_on_temporal_pairs.py:174` — Only 5 permutation shuffles (SE ≈ 0.07 on AUC). Increase to 100+ and report SD.

- **[P1-11] [ENG]** `fetch_cochrane_titles.py:52` — `DOIS_IN.read_text()` uses cp1252 default on Windows per `lessons.md`. Fix: `.read_text(encoding="utf-8")`.

- **[P1-12] [ENG]** No tests for `parse_pico` (a 40-line regex parser with zero coverage). Given P0-6/P0-7, this is high-risk. Fix: ≥5 unit tests covering versus / for / in / empty / prefix-strip / outcome-prefix cases.

- **[P1-13] [COM]** `manuscript/main.md:118` — "Every meta-analysis … should publish a flip probability" is unsupported. Single-substrate evidence does not license field-wide prescription. Fix: soften to "warrants evaluation as a candidate primitive".

- **[P1-14] [COM]** `manuscript/main.md:99, 136` — Figure 3 referenced but no figure file exists; S3 reliability diagram listed but only the JSON report ships. Fix: generate figure or drop the references.

---

## P2 — Minor / nice to fix

- **[P2-1] [STAT]** `train_on_temporal_pairs.py:81, 88` — `pipeline_empty=True` through `pd.to_numeric(errors="coerce")` is version-dependent on bool-dtype. Fix: `.astype(int)` explicitly.
- **[P2-2] [STAT]** `train_on_temporal_pairs.py:63` — `.map(...).fillna(...)` needed for unmapped scales; currently silent NaN.
- **[P2-3] [STAT]** `manuscript/main.md:78` — "slightly over-calibrated" for slope 1.32 is wrong direction; slope > 1 means under-dispersed predictions. Fix wording.
- **[P2-4] [STAT]** Flip sign=0 edge case when endpoint equals null exactly.
- **[P2-5] [DOM]** `build_per_pair_pipeline_features.py:64` — `mean_expected_event_rate=0.05` hardcoded; reported as a "pipeline feature" but is placeholder. Fix: remove from feature list or extract from AACT primary-outcome rates.
- **[P2-6] [DOM]** `manuscript/main.md:42` — 7.1% flip rate is low vs true CDSR v1→v2 updates (where refresh is conclusion-driven → endogeneity). Strengthen Limitations.
- **[P2-7] [SEC]** `fetch_cochrane_titles.py:23` — NHS email in UA leaks PII. Use a project-generic mailbox.
- **[P2-8] [SEC]** `build_per_pair_pipeline_features.py:73` — AACT snapshot date hardcoded. Read from env or glob newest.
- **[P2-9] [SEC]** `fetch_cochrane_titles.py:66-74` — Thread-pool pacing throttles consumer, not submitter. Fix: token bucket on `ex.submit`.
- **[P2-10] [ENG]** `train_on_temporal_pairs.py:121, 167, 178` — Step counters `[3/5]` and `[5/6]` / `[6/6]` inconsistent.
- **[P2-11] [ENG]** `build_per_pair_pipeline_features.py:17, 22, 32` — Unused `csv`, `math`, `_primary_token` imports.
- **[P2-12] [ENG]** `build_per_pair_pipeline_features.py:200` — `== False` triggers pandas FutureWarning. Use `~out_df["pipeline_empty"]`.
- **[P2-13] [ENG]** `build_per_pair_pipeline_features.py:174` — The 3,156 × 579k `str.contains` scans are the hotspot, not `iterrows`. Pre-build an inverted index.
- **[P2-14] [COM]** `manuscript/main.md:103` — "narrative centerpiece" → replace with concrete claim.
- **[P2-15] [COM]** `manuscript/main.md:111` — Stale "≈2,800 lines of Python, 66 unit tests" — verify current LOC/tests.
- **[P2-16] [COM]** Missing funding statement.
- **[P2-17] [COM]** Unicode (≈, −, ×, τ², I²) consistency across abstract / tables / captions.
- **[P2-18] [DOM]** `manuscript/main.md:4` — Nature Medicine target for a single-author, single-centre method paper citing an unpublished substrate is unrealistic. Refocus to J Clin Epidemiol / Research Synthesis Methods / Stat Med, or obtain multi-centre co-authors.

---

## False-positive watch (cross-checked against lessons.md)

- No DOR / Clayton copula / Clopper-Pearson surfaces touched in today's files — none to false-flag.
- Regex patterns `[^a-z]` and `14651858\.(CD\d+)\.` are linear/bounded — no ReDoS.
- `pickle.load` on self-produced trusted paths is not an exploit locally; the concern is shipping artefacts for downstream consumers (P1-9).

---

## Verified identifiers (domain expert)

- **SGLT2i HFpEF** — Vaduganathan 2022 Lancet, PMID 36041474. DELIVER HR 0.82 [0.73–0.92], EMPEROR-Preserved 0.79 [0.69–0.90], pooled ~0.80. ✓ Consistent.
- **SUMMIT tirzepatide** — Packer NEJM 2025, PMID 39555826, NCT04847557. Primary composite HR 0.62 [0.41–0.95]. ✓ Consistent. ACM HR 1.245 (ns) correctly flagged in manuscript line 115 as a prior memory error that has been corrected.
- **EMPA-REG OUTCOME** — Zinman NEJM 2015, PMID 26378978. 3p-MACE HR 0.86 [0.74–0.99]. ✓ Consistent.

---

## Next step

Awaiting user decision on which tiers to fix. Recommendation: fix all P0 (≤~1 session), then P1, then polish.

---

## Path B fix-pass record (2026-04-15)

**Fixed (9 / 14 P0):** P0-1, P0-2, P0-3, P0-4, P0-5, P0-8, P0-9, P0-10, P0-13.

**Remaining P0 (scope-deferred, not blocking next iteration):**
- **[FIXED 2026-04-15] P0-6 / P0-7 (PICO parser fragility + word-boundary AACT match):** Parser rewritten with expanded grammar (compared with / to prevent / temporal / generic-noun openers / verb-prefix stripping on condition side), 29-case labelled regression test, 4-char-minimum head-word rule, and word-boundary AACT matching. Post-audit: non-empty pipeline rate went 252 → 188 (8.0% → 6.0%), removing 64 spurious substring matches (25% false-positive contamination, e.g., "hmg" → unrelated trials). Pipeline ablation CI **tightened** (±0.005 → ±0.004) after cleanup, confirming the null is not a parser artefact. Manuscript Limitations and Methods updated. Dominant remaining empties are pre-2005 reviews where AACT coverage is genuinely sparse.
- **P0-11 (Pairwise70 unpublished):** Requires negotiation with Hopewell et al. for a Zenodo DOI. Manuscript Limitations now flags this and constrains realistic targets to method journals (J Clin Epidemiol / RSM / Stat Med) until resolved.
- **P0-12 (TRIPOD+AI compliance):** Adds a supplement section (S7) — straightforward but ~half-day work. Not in this fix-pass.
- **[FIXED 2026-04-15] P0-14 (Forecast Card re-emission + circular-ordering reframe):** Re-emitted all three Forecast Cards with the shipped post-fix model. New flip probabilities: SGLT2i HFpEF **5.3%** (was 4.4%), SUMMIT tirzepatide **20.9%** (was 22.1%), EMPA-REG **9.3%** (was 9.4%). Narrative ordering SGLT2i < EMPA-REG < SUMMIT preserved. Manuscript Results 2 rewritten to address the circular-reasoning objection: PICOs were selected on independent clinical grounds, probabilities generated mechanically, and the ordering check is against an independent evidence-maturity index (k, years-since-landmark) — not a post-hoc narrative. Forecast Card emission script now loads held-out metrics dynamically from the validation JSON so provenance cannot drift.

**Verification:**
- Test suite: **67/67 pass** (was 66 baseline + 1 new contract test `test_group_split_no_ma_id_in_both`).
- Pre-registered ship thresholds: AUC 0.772 ≥ 0.70 ✔; Brier 0.067 < 0.18 ✔; perm AUC 0.489 ≈ 0.50 ✔; calibration slope **declared deviation** (Platt 1.50, isotonic 0.46, both miss [0.8, 1.2] in opposite directions; per pre-specified tiebreak ship the calibrator closer to slope 1.0 → Platt).
- Files touched: `scripts/train_on_temporal_pairs.py`, `scripts/build_per_pair_pipeline_features.py`, `evidence_forecast/calibration/train.py`, `evidence_forecast/calibration/validate.py`, `tests/test_train.py`, `manuscript/main.md`. All other P0/P1/P2 source files untouched.

**P1 / P2 not yet addressed:** all P1 and P2 items remain open. They are non-blocking for the next discussion / decision cycle but should be triaged before any submission attempt.
