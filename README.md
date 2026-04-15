# Evidence Forecast

Probabilistic forecast of meta-analysis conclusion flips for cardiology PICOs.

## What it does

For any PICO, emits a **Forecast Card** — a cryptographically signed JSON + HTML bundle with:

1. **Effect**: current pooled estimate + 95% CI + prediction interval (CardioSynth)
2. **Flip probability**: P(conclusion reverses within 24 months) with bootstrap 95% CI
3. **Representativeness**: burden-weighted overlap between trial-country mix and disease burden (registry-first)
4. **TruthCert**: HMAC-SHA256 signature over the above

## Phase-1 seed PICOs

- **SGLT2i in HFpEF** — stability anchor (Vaduganathan 2022, Lancet)
- **Tirzepatide in HFpEF all-cause mortality** — tension-bearing (SUMMIT / Packer 2025)
- **Empagliflozin in T2DM CV outcomes** — calibration benchmark (EMPA-REG / Zinman 2015)

## Current status: parser-audited, calibration deviation declared, 12/14 P0 review findings resolved

121/121 tests pass (1 env-dependent skip). The released `outputs/` cards are signed JSON + HTML bundles with:
- **Real pooled effects** from native DL + HKSJ random-effects pool over source-verified study-level YAMLs (Vaduganathan 2022, SUMMIT Packer 2025, EMPA-REG Zinman 2015; all PMIDs, DOIs, NCTs cross-checked 2026-04-14).
- **Real calibrated flip-forecaster** trained on 1,669 analysed temporal pairs (3,156 raw pairs filtered by label-flips inclusion) extracted from 557 Cochrane review titles (560 `.rda` source files from the Pairwise70 collection). **Group-aware temporal split** by `ma_id` eliminates within-review cluster leakage. Held-out AUC 0.772, Brier 0.067, calibration slope 1.37, permutation AUC 0.488. Ship thresholds: AUC ≥ 0.70 PASS; Brier < 0.18 PASS; slope in [0.8, 1.2] **declared protocol deviation** — both isotonic (0.46) and Platt (1.37) miss in opposite directions; ship Platt per pre-specified closer-to-1.0 tiebreak.
- **Real per-pair AACT pipeline features** from the canonical 2026-04-12 extract (~580k trials joined into `cache/aact_joined_2026-04-12.csv`), with expanded Cochrane-title parser + word-boundary AACT matching after a 29-case labelled parser audit. Pipeline-features ablation: **ΔAUC +0.0007, 95% paired-bootstrap CI [−0.0036, +0.0052], n_boot=1,000** — the pre-registered pipeline-conditioning novelty hypothesis is rigorously null at the measurable scale.
- **TruthCert HMAC-SHA256 signature** over canonical JSON, key drawn from `TRUTHCERT_HMAC_KEY` env var only, constant-time compare.

### What remains before submission

1. **Pairwise70 Zenodo DOI** — the substrate is an unpublished Hopewell et al. collection; a citable DOI for the analysed subset is needed before tier-1 venue submission.
2. **TRIPOD+AI 2024 supplement** — mandatory for prediction-model papers at reputable methods venues; ~half-day standalone task.
3. **Registry-first population weights** into `scripts/run_forecast.py::_load_weights` for non-zero representativeness scores. Integrates with the sister project `registry_first_rct_meta`.
4. **True Cochrane CDSR version-pair integration** — requires Wiley institutional API access; see `REAL_PHASE1_BLOCKERS.md` option 2.

See `review-findings.md` for the 14 P0 / 14 P1 / 18 P2 multi-persona review and the resolution record.

## Usage

Paths on your machine are resolved via env vars (no literal drive letters required):

```bash
# Required at session start
export TRUTHCERT_HMAC_KEY="your-secret-key"     # HMAC key, never committed

# Optional — only if AACT or Pairwise70 live somewhere non-default. The
# discovery helpers evidence_forecast._aact_paths and scripts/_pairwise70_paths.R
# check env vars first, then fall back to candidate roots.
export AACT_ROOT="/path/to/AACT/2026-04-12"     # raw tables directory
export AACT_PATH="$AACT_ROOT/studies.txt"       # or the joined CSV
export PAIRWISE70_ROOT="/path/to/Pairwise70/data"
```

One-time pipeline to reproduce cache + trained model:
```bash
# 1. Build AACT joined cache (~5 min) — reads from AACT_ROOT or candidates
python scripts/build_aact_cache.py
# 2. Extract DOIs, fetch Cochrane titles from CrossRef, build per-pair pipelines
python scripts/fetch_cochrane_titles.py
python scripts/build_per_pair_pipeline_features.py
# 3. Extract real temporal Cochrane pairs (~3 min, requires R + metafor)
Rscript scripts/extract_temporal_pairs.R
# 4. Train flip-forecaster with isotonic + Platt calibration comparison
python scripts/train_on_temporal_pairs.py
```

Emit cards:
```bash
# Using the real trained model (dynamic provenance from validation JSON):
python scripts/generate_trio_real_model.py
# Single-PICO CLI:
python scripts/run_forecast.py --pico sglt2i_hfpef
```

## Static vs dynamic components

| Item | Source | Static / Dynamic |
|---|---|---|
| 3 seed PICOs | `configs/picos/*.yaml` | Static (hand-curated) |
| AACT snapshot | resolved via `AACT_ROOT` env var / `evidence_forecast._aact_paths` | Static per Phase-1 release (2026-04-12) |
| Pairwise70 substrate | resolved via `PAIRWISE70_ROOT` env var / `scripts/_pairwise70_paths.R` | Static |
| IHME burden | `gbd2023_all_burden_204countries_1990_2023.parquet` | Static (pending wire-up) |
| Pooled effect | native DL+HKSJ (`evidence_forecast._native_pool`); `configs/studies/*.studies.yaml` | Dynamic (recomputed per run) |
| Flip model | `models/flip_forecaster_v1.pkl` (GBM + Platt, group-split temporal pairs) | Static (retrainable via `scripts/train_on_temporal_pairs.py`) |
| Pipeline features | AACT joined CSV matched at each pair's `v1_date` with word-boundary regex | Dynamic (per snapshot date + v1 date) |
| TruthCert signature | computed per run from `TRUTHCERT_HMAC_KEY` env var | Dynamic |

## Validation

See `models/validation_report_temporal_calibrated.json` (shipped Platt) and `models/validation_report_temporal_isotonic.json` (sensitivity). Held-out metrics on n_test = 1,107:
- **AUC 0.772** (threshold ≥ 0.70: PASS)
- **Brier 0.067** (threshold < 0.18: PASS)
- **Calibration slope 1.37** (threshold [0.8, 1.2]: **declared protocol deviation**)
- **Permutation AUC 0.488** (should be ≈ 0.50: PASS — no rank leakage)
- **Pipeline-features ablation ΔAUC +0.0007, 95% bootstrap CI [−0.0036, +0.0052]** (n_boot = 1,000)

Training substrate: 1,669 analysed temporal pairs (3,156 raw, filtered by label-flips complete-CI + 6–48 month gap) from Pairwise70 `.rda` files, constructed by pooling study-level data at the 55th and 80th Study.year percentiles via `metafor::rma(method="DL")`. 7.2% base flip rate. 557 / 560 Cochrane review titles fetched via CrossRef for per-pair AACT enrichment; 188 / 3,156 pairs (6.0%) receive a non-empty AACT pipeline at v1-date after word-boundary matching.

## Architecture

Four-layer pipeline:

```
PICO.yaml
  ├─► effect_layer          ─┐
  ├─► pipeline_layer         ─┤
  ├─► representativeness     ─┼─► forecast_card ─► truthcert_layer ─► signed bundle
  └─► flip_forecaster ───────┘
         ↑
   models/flip_forecaster_v1.pkl
```

Imports from (does not duplicate) CardioSynth, registry-first-rct-meta, EvidenceOracle, Overmind TruthCert.

## Non-goals (Phase 1)

- No atlas UI. No monthly auto-refresh cron. No NICE recommendations (Phase 2).
- No re-implementation of CardioSynth / registry-first / EvidenceOracle / Overmind — imports only.

## Citation

Manuscript draft in `manuscript/main.md`. Target: Nature Medicine.

Author: Mahmood Ahmad, Tahir Heart Institute (ORCID 0009-0003-7781-4478).
