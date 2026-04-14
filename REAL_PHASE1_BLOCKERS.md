# Real Phase-1 Blockers (investigated 2026-04-14)

The plan at `C:\Users\user\docs\superpowers\plans\2026-04-14-evidence-forecast-plan.md` Task 17 assumed two data prereqs that on investigation don't exist on disk in the form the plan required. This file records what was found so the next session doesn't redo the investigation.

## Blocker 1: CardioSynth has no Python API

**Plan assumed**: `C:\cardiosynth\cardiosynth\engine.py::pool_for_pico(...)` returning a `MetaAnalysisResult`-like object.

**Reality**: `C:\cardiosynth\` is a client-side HTML app.
- `core/metaengine.html`, `core/truthcert.html`, `core/update-engine.html`, `core/provenance-store.html`, `core/disagreement-queue.html`
- `synthesis/bias-quantifier.html`, `synthesis/portfolio-aggregator.html`
- `harvesters/`, `validators/`, `phase0/` (not yet inspected — may contain Python but not at the signature the plan assumed)

**Resolution options** (increasing effort):
1. **Inspect `harvesters/`, `validators/`, `phase0/`** for any Python that computes pooled effects. If found, point the adapter at it.
2. **Write a small Python pooling module** (`evidence_forecast/_native_pool.py`) using `statsmodels` / inverse-variance DL + HKSJ. ~150 LoC. Use it as the default backend. CardioSynth becomes a comparison tool, not a dependency.
3. **Call CardioSynth via headless browser** (Selenium / Playwright). Heavy-weight; only if exact CardioSynth numbers are needed for traceability.

**Recommended**: option 2. Native pooling is a well-defined ~150 LoC module with R/metafor validation tests available via `advanced-stats.md` rules. Decouples Evidence Forecast from a UI that evolves independently.

## Blocker 2: No multi-version pair dataset exists on disk

**Plan assumed**: `C:\MetaAudit\outputs\pairs.csv` with columns `ma_id, v1_date, v2_date, outcome, v1_point, v1_ci_low, v1_ci_high, v2_point, v2_ci_low, v2_ci_high, topic_area, scale`.

**Reality**: MetaAudit audited **one pub per review** — 473 unique review-pub combinations in `C:\MetaAudit\results\audit_results.csv`, with **zero reviews having ≥2 pub versions captured**. `ma_id` format is `CD######_pubN_data__A#` and each review appears with its single latest pub only.

Related data found:
- `C:\Projects\Pairwise70\analysis\ma4_results_pairwise70.csv`: 5,089 rows, 85 unique reviews, each at a single pub version. Rich effect-size fields (`theta, ci_std_lo, ci_std_hi, tau2, I2`) but not longitudinal.
- `audit_results.csv`: 68,520 audit-module findings. No effect sizes.

**Resolution options**:
1. **Acquire Cochrane historical versions**: Cochrane's CDSR API returns prior review versions with their effect sizes. Requires authenticated access. A scraper over the Cochrane website is fragile and likely ToS-problematic.
2. **Use retractions / conclusion-change datasets**: `C:\Users\user\retraction-gravity\`, `C:\Users\user\evidence-collapsar\`, and `C:\Models\MetaShift\` may contain longitudinal conclusion-change records. Investigate before building from scratch.
3. **Synthesise pairs from simulation**: generate plausible v1 effect + CI + pipeline features, simulate v2 after trial arrival, label flip. Weakens the paper to "if reality matches these simulation assumptions" but is fast.
4. **Refocus the paper** on the primitive itself (architecture + flip-label contract + pipeline-feature family + cryptographic attestation) and defer empirical calibration to a follow-up once longitudinal data is acquired. This is the most honest path for Phase 1.

**Recommended**: option 4 as the immediate Phase-1 completion path; option 2 investigation in parallel as a candidate data source for the empirical Phase-1.5.

## What already works (no blocker)

- Four-layer architecture, all modules unit-tested (58/58 tests passing).
- TruthCert HMAC-SHA256 signing with env-var-only key, constant-time compare.
- Pipeline-features family extracts Shannon entropy, design heterogeneity, Σ expected N from AACT fixture (canonical AACT snapshot at `C:\Users\user\AACT\2026-04-12\` is present and confirmed by `INDEX.md`).
- Dev-mode bootstrap produces three signed Forecast Cards for SGLT2i HFpEF, SUMMIT tirzepatide ACM, EMPA-REG T2DM anchored to published primary MAs.
- JSON schema contract + module-boundary contract tests enforce MetaReproducer-P0-1-style field-name stability.

## The next commit after this file

Write `evidence_forecast/_native_pool.py` (resolution option 2 for Blocker 1) to unblock real effects even before real flip-model training lands. This makes the dev-mode "stub" effects replaceable with native Python pooling, which is a meaningful improvement regardless of how Blocker 2 resolves.
