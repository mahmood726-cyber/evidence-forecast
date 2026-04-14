# E156 Protocol — Evidence Forecast

**Project**: Evidence Forecast (Phase 1)
**Repo**: github.com/mahmood726-cyber/evidence-forecast
**Dates**: started 2026-04-14
**Dashboard**: https://mahmood726-cyber.github.io/evidence-forecast/

## CURRENT BODY

Will meta-analytic conclusions flip? We built a four-layer Forecast Card — pooled effect with prediction interval, calibrated flip probability, burden-weighted representativeness, and HMAC-SHA256 cryptographic attestation — and trained the flip layer on 1,632 real temporal version pairs extracted from 560 Cochrane study-level datasets by pooling each analysis's studies at year percentiles via `metafor::rma(method="DL")`. Method: gradient-boosted classifier with post-hoc Platt sigmoid calibration over a temporal split (v1 < 2015 train, v1 ≥ 2015 test, n_test 1,080). Result: held-out AUC 0.777, Brier 0.064, calibration slope 1.32; label-permutation AUC 0.514 confirms no leakage. Trio forecast: SGLT2i HFpEF flip 4.4% (Vaduganathan 2022 pool), SUMMIT tirzepatide 22.1% (single-trial tension-bearing), EMPA-REG 9.4% (replicated CVOT). Interpretation: conclusion-flip probability is learnable from real Cochrane data even without pipeline conditioning and can be emitted as a signed, auditable primitive per PICO. Boundary: training uses within-review temporal accretion rather than true cross-version updates; pipeline-features family is declared but not yet empirically contributed to AUC; Cochrane CDSR integration is Phase-1.5.

## YOUR REWRITE



## SUBMITTED: [ ]
