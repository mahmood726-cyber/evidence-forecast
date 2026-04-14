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

## Current status: DEV MODE

Phase-1 Python scaffolding is complete with 58/58 tests passing. The released `outputs/` cards are **dev-mode artifacts** — signed and functional but generated from:
- Stub effects anchored to published primary MAs (CardioSynth integration pending)
- Synthetic-data flip-forecaster (real MetaAudit-trained model pending)
- Fixture AACT subset (canonical extract wire-up pending)
- Empty representativeness (registry-first population layer integration pending)

**To ship real Phase-1** (user-present Task 17):
1. Align `evidence_forecast/_cardiosynth_adapter.py` with the actual CardioSynth engine API on this machine.
2. Produce or point to `C:\MetaAudit\outputs\pairs.csv` with columns `ma_id, v1_date, v2_date, outcome, v1_point, v1_ci_low, v1_ci_high, v2_point, v2_ci_low, v2_ci_high, topic_area, scale` plus the v1-snapshot feature columns documented in `evidence_forecast/calibration/features.py`.
3. Run `python scripts/train_calibration.py --pairs <path> --aact C:/Users/user/AACT/2026-04-12/studies.txt --out ./models`. Ship thresholds: AUC ≥ 0.70, Brier < 0.18, calibration slope in [0.8, 1.2].
4. Wire registry-first population weights into `scripts/run_forecast.py::_load_weights` for real representativeness scores.
5. Cross-check each PICO card against the primary-MA paper (NCT, PMID, DOI, effect magnitudes and dates).
6. Regenerate with `python scripts/generate_trio_cards.py` (non-dev path).

## Usage

```bash
# Dev mode (no external deps):
export TRUTHCERT_HMAC_KEY="your-secret-key"
python scripts/dev_bootstrap.py
# outputs/*.json + outputs/*.html

# Production mode (requires CardioSynth + trained model):
python scripts/run_forecast.py --pico sglt2i_hfpef
python scripts/generate_trio_cards.py
```

## Static vs dynamic components

| Item | Source | Static / Dynamic |
|---|---|---|
| 3 seed PICOs | `configs/picos/*.yaml` | Static (hand-curated) |
| AACT snapshot | `C:\Users\user\AACT\2026-04-12` | Static per Phase-1 release |
| IHME burden | `gbd2023_all_burden_204countries_1990_2023.parquet` | Static (pending wire-up) |
| CardioSynth effect | recomputed per run (pending adapter match) | Dynamic |
| Flip model | `models/flip_forecaster_v1.pkl` | Static (trained once) |
| TruthCert signature | computed per run | Dynamic |

## Validation

See `models/validation_report_v1.json`. Ship thresholds: AUC ≥ 0.70, Brier < 0.18, calibration slope in [0.8, 1.2]. Current dev-mode values come from a synthetic training set and are not release-grade.

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
