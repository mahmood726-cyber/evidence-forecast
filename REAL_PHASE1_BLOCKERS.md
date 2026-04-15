# Real Phase-1 Blockers (investigated 2026-04-14)

The plan at `<user-docs>/superpowers/plans/2026-04-14-evidence-forecast-plan.md` Task 17 assumed two data prereqs that on investigation don't exist on disk in the form the plan required. This file records what was found so the next session doesn't redo the investigation.

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

## Blocker 2: Real multi-version pair dataset is n=6 (seed only)

**Plan assumed**: `C:\MetaAudit\outputs\pairs.csv` with ~1,500–2,500 pairs.

**Reality after investigation**: MetaAudit audited one pub per review (473 unique review-pubs, zero multi-version). Pairwise70 has 595 `.rda` files covering 591 unique reviews × 1 pub each plus **4 reviews with 2 pubs each**. These 4 multi-pub reviews yielded **6 real pair rows** after pooling each analysis via metafor DL random-effects (`scripts/extract_cochrane_pairs.R`, fixture at `tests/fixtures/real_cochrane_pairs_v0.csv`).

All 6 real pairs are flip=0 in the current seed — useful as sanity anchors for the contract (verified by `tests/test_real_cochrane_pairs.py`), but insufficient for training a real flip-forecaster.

No other candidate data sources found on disk:
- `retraction-gravity`, `evidence-collapsar`: speculative simulation tools, not longitudinal MA data.
- `MetaShift`: not present on disk (memory pointer stale).
- `Pairwise70/analysis/ma4_results_pairwise70.csv`: 85 unique reviews at single-pub each.

**Resolution options** (increasing effort / payoff):
1. **Synthetic-but-grounded pair generator**: use Pairwise70's `ma_summary.csv` distributions (85 real MAs × tau² × I²) to simulate plausible v1→v2 trajectories with realistic flip rates. Train on ~1,000 synthetic pairs; external-validate on the 6 real pairs. Weakens the paper to "if reality matches these simulation assumptions" but is shippable in Phase-1.
2. **Cochrane CDSR API acquisition**: requires Wiley institutional access. Pulls historical review versions with effect sizes. 2–4 days to scaffold and validate.
3. **Manual curation from published updates**: read ~200 Cochrane PDFs with multiple versions, extract effect-size changes by hand. Weeks of work.
4. **Refocus the paper as primitive + 6-pair retrospective case study**: honest BMJ Analysis / J Clin Epidemiol paper — "here's the flip-forecast primitive; here's how it would have scored these 6 real retrospective updates." Does not require external data. Smaller venue than Nature Med, shippable now.

**Recommended** (as of 2026-04-14): option 4 for immediate shipping; option 1 for Phase-1.5; option 2 if institutional access materialises.

## What already works (no blocker)

- Four-layer architecture, all modules unit-tested (58/58 tests passing).
- TruthCert HMAC-SHA256 signing with env-var-only key, constant-time compare.
- Pipeline-features family extracts Shannon entropy, design heterogeneity, Σ expected N from AACT fixture (canonical AACT snapshot discovered via `evidence_forecast._aact_paths.discover_root()`; see `aact_storage_location` memory for the active location).
- Dev-mode bootstrap produces three signed Forecast Cards for SGLT2i HFpEF, SUMMIT tirzepatide ACM, EMPA-REG T2DM anchored to published primary MAs.
- JSON schema contract + module-boundary contract tests enforce MetaReproducer-P0-1-style field-name stability.

## The next commit after this file

Write `evidence_forecast/_native_pool.py` (resolution option 2 for Blocker 1) to unblock real effects even before real flip-model training lands. This makes the dev-mode "stub" effects replaceable with native Python pooling, which is a meaningful improvement regardless of how Blocker 2 resolves.
