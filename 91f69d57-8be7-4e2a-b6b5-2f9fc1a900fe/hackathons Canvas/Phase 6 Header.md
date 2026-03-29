# Phase 6 — Business Recommendation

**Segmenting 3,033 users into 3 retention tiers using calibrated GBM probabilities (p_retain) and SHAP explainability from Phase 5.**

| Block | Purpose |
|-------|---------|
| **GBM Retention Classifier** (Phase 4) | Calibrated model + 5-fold CV metrics |
| **SHAP Analysis** (Phase 5) | Feature importance on test set |
| **Business Recommendation** (Phase 6) | Tier segmentation + executive strategy |

> 📊 Tier thresholds: **High Risk** (p < 0.30) · **At Risk** (0.30–0.60) · **Healthy** (p ≥ 0.60)
