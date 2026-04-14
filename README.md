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

## Current status: real-data trained, submission-track

66/66 tests pass. The released `outputs/` cards are signed JSON + HTML bundles with:
- **Real pooled effects** from native DL + HKSJ random-effects pool over source-verified study-level YAMLs (Vaduganathan 2022, SUMMIT Packer 2025, EMPA-REG Zinman 2015; all PMIDs, DOIs, NCTs cross-checked 2026-04-14).
- **Real calibrated flip-forecaster** trained on 1,632 temporal version pairs extracted from 560 Cochrane study-level datasets (`tests/fixtures/temporal_cochrane_pairs_v0.csv`). Held-out AUC 0.777, Brier 0.064, calibration slope 1.32, permutation AUC 0.514. Ship thresholds: AUC ≥ 0.70 PASS; Brier < 0.18 PASS; slope in [0.8, 1.2] near-miss.
- **Real AACT pipeline features** from the canonical 2026-04-12 extract (579,828 trials joined into `cache/aact_joined_2026-04-12.csv`).
- **TruthCert HMAC-SHA256 signature** over canonical JSON, key drawn from `TRUTHCERT_HMAC_KEY` env var only, constant-time compare.

### What remains before Nature Med submission

1. **Pipeline-features enrichment per training pair** — currently filled with neutral constants in training (feature families 3–6). Pipeline-ablation ΔAUC (the paper's declared novelty claim) cannot be reported until per-pair PICO → AACT matching is wired for the 345 unique Cochrane reviews in the training set. AUC 0.777 is therefore a *lower bound* reflecting only effect geometry + heterogeneity.
2. **Registry-first population weights** into `scripts/run_forecast.py::_load_weights` for non-zero representativeness scores. Integrates with the sister project `C:\Projects\registry_first_rct_meta\`.
3. **True Cochrane CDSR version-pair integration** — requires Wiley institutional API access; see `REAL_PHASE1_BLOCKERS.md` option 2.

See `REAL_PHASE1_BLOCKERS.md` for the honest accounting and four resolution paths.

## Usage

One-time setup (reproduces the cache + real-trained model):
```bash
export TRUTHCERT_HMAC_KEY="your-secret-key"     # HMAC key, never committed
# Build AACT joined cache (~5 min, requires C:/Users/user/AACT/2026-04-12/):
python scripts/build_aact_cache.py
# Extract real temporal Cochrane pairs (~3 min, requires R + metafor):
Rscript scripts/extract_temporal_pairs.R
# Train flip-forecaster with Platt calibration:
python scripts/train_on_temporal_pairs.py
```

Emit cards:
```bash
# Using the real trained model:
python scripts/generate_trio_real_model.py
# Or for ad-hoc / reproducibility checks with stub model:
python scripts/dev_bootstrap.py
```

## Static vs dynamic components

| Item | Source | Static / Dynamic |
|---|---|---|
| 3 seed PICOs | `configs/picos/*.yaml` | Static (hand-curated) |
| AACT snapshot | `C:\Users\user\AACT\2026-04-12` | Static per Phase-1 release |
| IHME burden | `gbd2023_all_burden_204countries_1990_2023.parquet` | Static (pending wire-up) |
| Pooled effect | native DL+HKSJ (`evidence_forecast._native_pool`); `configs/studies/*.studies.yaml` | Dynamic (recomputed per run) |
| Flip model | `models/flip_forecaster_v1.pkl` (GBM + Platt, temporal pairs) | Static (retrainable via `scripts/train_on_temporal_pairs.py`) |
| Pipeline features | `cache/aact_joined_2026-04-12.csv` from canonical AACT extract | Dynamic (per snapshot date) |
| TruthCert signature | computed per run from `TRUTHCERT_HMAC_KEY` env var | Dynamic |

## Validation

See `models/validation_report_temporal_calibrated.json`. Held-out metrics on n_test=1,080:
- **AUC 0.777** (threshold ≥ 0.70: PASS)
- **Brier 0.064** (threshold < 0.18: PASS)
- **Calibration slope 1.32** (threshold [0.8, 1.2]: near-miss, 10% over upper)
- **Permutation AUC 0.514** (should be ≈ 0.50: PASS — no leakage)

Training substrate: 1,632 real temporal pairs from Pairwise70 `.rda` files, constructed by pooling study-level data at the 55th and 80th Study.year percentiles via `metafor::rma(method="DL")`. 345 unique Cochrane reviews, 7.1% base flip rate.

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
