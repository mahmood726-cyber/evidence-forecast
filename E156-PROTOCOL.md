# E156 Protocol — Evidence Forecast

**Project**: Evidence Forecast (Phase 1)
**Repo**: github.com/mahmood726-cyber/evidence-forecast
**Dates**: started 2026-04-14
**Dashboard**: https://mahmood726-cyber.github.io/evidence-forecast/

## CURRENT BODY

> Draft — Phase 1 dev-mode. Rewrite after real-data Task 17 with MetaAudit-trained AUC and Brier filled in.

Will meta-analytic conclusions flip? We built a flip-probability forecaster with four layers — pooled effect, pipeline-conditioned flip risk, burden-weighted representativeness, and cryptographic attestation — and applied it to three cardiology PICOs. Data: a calibration pipeline over MetaAudit version pairs (6–48-month gaps) with a novel pipeline-features family summarising ongoing registered trials at the v1 snapshot. Method: XGBoost/GBM with temporal train/test split at 2023-01-01 and cardiology fully held out from training. Result: synthetic-substrate held-out AUC 0.59 in dev mode; real-data AUC pending MetaAudit pair export; the SUMMIT tirzepatide ACM PICO emits a 74% dev-mode flip probability driven by ongoing ACM-designed trials. Robustness: permutation shuffling collapses AUC to chance as expected; an L1-logistic comparator is persisted for interpretability. Interpretation: conclusion-flip probability is operationalisable as a signed forecast card — the primitive is built; calibration quality gates Phase-1 ship. Boundary: 24-month horizon, binary CI-crosses-null flip label, cardiology held out for Phase-2 NICE generalisation.

## YOUR REWRITE



## SUBMITTED: [ ]
