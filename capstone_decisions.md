# Capstone Design Decisions (Detailed Rationale)

## Table of Contents
- [Problem Framing](#problem-framing)
- [Data Handling & Preprocessing](#data-handling--preprocessing)
- [Feature Engineering Strategy](#feature-engineering-strategy)
- [Modeling & Forecasting Approach](#modeling--forecasting-approach)
- [Evaluation Methodology](#evaluation-methodology)
- [Production Readiness & Automation](#production-readiness--automation)
- [Risk Management & Robustness](#risk-management--robustness)
- [Tooling & Code Organization](#tooling--code-organization)
- [Documentation & Reproducibility](#documentation--reproducibility)
- [Healthcare Domain Specific Decisions](#healthcare-domain-specific-decisions)
- [Pipeline Architecture Decisions](#pipeline-architecture-decisions)
- [Model Deployment & Monitoring](#model-deployment--monitoring)

---

## Problem Framing

### 1. DQ_SCORE Calculation Methodology and Interpretation
**Decision:** Use a heuristic-based DQ_SCORE calculated from test results, severity levels, and trust metrics.

**DQ_SCORE Formula:**
```
DQ_SCORE = (Number of Passed Tests / Total Number of Tests) per measurement period
```

**Components:**
- **Passed Tests**: Count of tests with STATUS = 'PASS' in a measurement period
- **Total Tests**: Count of all tests executed in the same measurement period
- **Score Range**: 0.0 to 1.0 (where 1.0 = perfect quality, 0.0 = complete failure)

**Calculation Example:**
```
Measurement Period: 2024-01-15
Total Tests Executed: 25
Passed Tests: 22
Failed Tests: 3

DQ_SCORE = 22 / 25 = 0.88
```

**DQ_SCORE Interpretation:**
- **0.90-1.00**: Excellent data quality (90-100% pass rate)
- **0.80-0.89**: Good data quality (80-89% pass rate)
- **0.70-0.79**: Acceptable data quality (70-79% pass rate)
- **0.60-0.69**: Poor data quality (60-69% pass rate)
- **Below 0.60**: Critical data quality (less than 60% pass rate)

**Business Meaning:**
- **Higher scores** indicate reliable, trustworthy data for downstream systems
- **Lower scores** signal potential data risks requiring immediate attention
- **Score trends** show data quality improvement or degradation over time
- **Score volatility** indicates system stability or underlying issues

**Why Simple Pass Rate Approach:**
- **Simplicity**: Easy to calculate and understand across all stakeholders
- **Standardization**: Consistent metric that can be compared across systems and time periods
- **Transparency**: Clear relationship between test execution and quality score
- **Actionability**: Directly tied to test improvement initiatives
- **Normalization**: 0-1 scale allows for easy aggregation and comparison

### 2. Forecasting Target: Next Measurement DQ_SCORE
**Decision:** Frame the task as “forecast the next measurement’s DQ_SCORE” rather than predicting the same-day score.

**Rationale:**
- In production, you typically want to know “what will the next run look like?” before it happens.
- Predicting the same-day score can leak information (features derived from that day’s outcomes).
- Aligns with real-world decision-making: you have today’s results and want to anticipate the next run.

**Alternatives Considered:**
- Predict same-day DQ_SCORE: rejected due to leakage.
- Predict calendar-date-specific score: rejected because measurements are irregular; you can’t predict a date with no run.
- Predict multiple steps ahead: rejected due to tiny dataset (28 rows) and lack of stable horizon.

---

## Data Handling & Preprocessing

### 2. Use Irregular Time Series Dataset
**Decision:** Use `feature_engineered_events_irregular.csv` (AI-agent-ready) instead of the standard engineered dataset.

**Rationale:**
- The raw measurements are irregular in time (gaps of days to weeks).
- Standard engineered features (rolling windows, fixed lags) assume regular intervals and can be misleading.
- The irregular processor removes misleading regular-interval features and adds gap-aware features (e.g., days since last measurement, measurement frequency).

**Impact:**
- More realistic for production where measurements don’t arrive on a fixed schedule.
- Avoids infinite/NaN values caused by rolling windows on sparse data.

### 3. Generate `events.csv` from `pseudo_deident.csv`
**Decision:** In the notebook, transform `pseudo_deident.csv` into enhanced `events.csv` with DQ_EVENT JSON structures.

**Rationale:**
- The preprocessing pipeline expects `events.csv` as input with enhanced features
- Raw `pseudo_deident.csv` lacks the DQ_EVENT JSON structures and additional metrics
- Transformation adds healthcare-specific parsing and feature enrichment
- Creates the comprehensive dataset needed for advanced feature engineering

### 4. Keep Only Numeric Features for Modeling
**Decision:** Filter to numeric-only features before training/forecasting.

**Rationale:**
- The dataset contains categorical derivatives (`DQ_SCORE_CATEGORY`, `TRUST_SCORE_CATEGORY`) that are redundant with the numeric scores.
- Simplifies modeling and avoids encoding overhead for a tiny dataset.
- Ensures compatibility with scikit-learn models without additional preprocessing.

---

## Feature Engineering Strategy

### 5. Exclude Future-Leaking Columns
**Decision:** Remove columns like `days_to_next_measurement` and any `*_to_next` fields before modeling.

**Rationale:**
- These columns contain information about the future (when the next measurement occurs).
- In a real forecast, you don’t know the future gap until the next run happens.
- Prevents artificially optimistic performance.

### 6. Mode-Dependent Feature Sets (post_run vs pre_run)
**Decision:** Provide a `mode` flag in the production module:
- `post_run`: include daily aggregates (violations, failures) that are known after a run completes.
- `pre_run`: exclude those aggregates to simulate forecasting before a run.

**Rationale:**
- In production, you may forecast either after a run (when you know that day’s outcomes) or before a run (when you don’t).
- Gives flexibility without maintaining two separate pipelines.
- Aligns with real-world deployment timing.

---

## Modeling & Forecasting Approach

### 7. Include Naive(last_value) as a Baseline
**Decision:** Always evaluate a “next ≈ current” baseline alongside ML models.

**Rationale:**
- On small, stable time series, the naive baseline is often the best forecast.
- Provides a minimum performance bar; if ML can’t beat it, don’t deploy it.
- Transparent and easy to explain to stakeholders.

**Formulation of the Naive(last_value) Model:**
- **Mathematical Definition:** ŷ_{t+1} = y_t
  - Where ŷ_{t+1} is the forecast for the next measurement
  - y_t is the last observed DQ_SCORE value
- **Implementation in Walk-Forward:**
  - For each test date t, the prediction is simply the DQ_SCORE from the most recent previous measurement
  - No training is required; the model "learns" nothing
  - Handles irregular time gaps naturally (e.g., if the last measurement was 7 days ago, that value is used)
- **Why It's Considered "Naive":**
  - Ignores all engineered features (daily counts, violations, gaps)
  - Assumes the series has no trend or seasonality
  - Represents the simplest possible time-series forecast
  - Serves as a benchmark: any ML model must meaningfully outperform this to justify complexity

### 7a. Why "Last DQ_SCORE ≈ Next DQ_SCORE" Emerged as Best
**Decision:** Accept and document that the naive baseline (last observed DQ_SCORE) consistently outperformed ML models.

**Rationale:**
- **Data Characteristics:** With only 28 measurements and high day-to-day stability, the series exhibits low volatility. When a metric is stable, the best predictor of the next value is often the current value.
- **Insufficient Signal:** The engineered features (daily counts, violation flags, gap metrics) did not contain enough predictive signal to overcome the simplicity of "tomorrow ≈ today."
- **Overfitting Risk:** ML models attempted to fit noise rather than signal due to the tiny sample size, leading to worse generalization than the baseline.
- **Temporal Consistency:** The DQ_SCORE represents a data quality index that typically evolves slowly unless there are major changes in data pipelines or schemas. In a stable production environment, such changes are infrequent.
- **Empirical Evidence:** Walk-forward validation showed the naive baseline achieved the lowest RMSE, confirming that for this specific dataset and context, the naive forecast is the most reliable choice.

### 8. Candidate Models: Shallow Tree + Regularized Linear Models
**Decision:** Evaluate:
- DecisionTree(max_depth=2, min_samples_leaf=2)
- ElasticNet (alpha=0.1, l1_ratio=0.5)
- BayesianRidge
- Ridge (alpha=5.0)

**Rationale:**
- **DecisionTree(depth=2):** Interpretable, handles non-linear patterns, but shallow to avoid overfitting on 28 rows.
- **ElasticNet:** Handles multicollinearity, performs feature selection via L1.
- **BayesianRidge:** Probabilistic, often robust with small data.
- **Ridge:** Strong regularization (alpha=5.0) for small datasets.

**Excluded:**
- Deep trees, ensembles, boosting: too much variance for 28 rows.
- Unregularized linear regression: overfits with many features.

### 9. Walk-Forward (Expanding Window) Validation
**Decision:** Use walk-forward validation instead of random train/test split.

**Rationale:**
- Preserves temporal order; no future leakage.
- Simulates production: train on past, predict next.
- Provides multiple test points (important with tiny data).

**Implementation Details:**
- Minimum training size: 8 observations (or 40% of data, whichever is larger).
- Each step trains on all available data up to time t, predicts t+1.
- RMSE computed across all predicted steps.

### 10. Automatic Model Selection in Production
**Decision:** Implement `train_and_forecast_next` to automatically:
- Run walk-forward evaluation on all candidates
- Select the model with lowest RMSE
- Retrain on all labeled data
- Forecast the next measurement
- Save an artifact with model, scaler, feature list, and metrics

**Rationale:**
- In production, you can’t manually re-evaluate models each time.
- As new data arrives, the best model may change; automation handles this.
- Artifact persistence enables reproducibility and auditability.

---

## Evaluation Methodology

### 11. Primary Metric: RMSE
**Decision:** Use Root Mean Squared Error (RMSE) as the ranking metric.

**Rationale:**
- Penalizes larger errors more heavily (important for quality scores).
- Interpretable in the same units as DQ_SCORE.
- Standard in forecasting literature.

**Secondary Metrics (logged but not used for selection):**
- MAE (less sensitive to outliers)
- R² (for explanatory purposes, but unstable with tiny data)

### 12. No Hold-Out Test Set
**Decision:** Do not hold out a final test set; use all data for training after model selection.

**Rationale:**
- With only 28 rows, a hold-out would leave too few for training.
- Walk-forward already provides a realistic performance estimate.
- The artifact saves the evaluation metrics for audit.

---

## Production Readiness & Automation

### 13. Save Timestamped Artifacts
**Decision:** Save model artifacts with a UTC timestamp (e.g., `dq_score_next_forecaster_20260104_233900.pkl`).

**Rationale:**
- Enables versioning and rollback.
- Prevents accidental overwrites.
- Facilitates audit trails.

### 14. Persist Feature List and Scaler
**Decision:** Include feature column names and scaler (if used) in the saved artifact.

**Rationale:**
- In production, you must ensure the same feature order and scaling at prediction time.
- Avoids “feature drift” between training and serving.
- Makes the artifact self-contained.

### 15. Provide a Minimal, Reusable Notebook
**Decision:** Create `Capstone_notebook.ipynb` with only the essential steps.

**Rationale:**
- Reduces cognitive load for reviewers and stakeholders.
- Easier to maintain and debug.
- Clear separation between experimentation (original notebook) and production flow.

---

## Risk Management & Robustness

### 16. Prefer Naive Baseline with Tiny Data
**Decision:** If the naive baseline wins, recommend using it in production.

**Rationale:**
- Simpler models are more robust to distribution shifts.
- With 28 rows, ML models can overfit noise.
- Naive baseline is transparent and always available.

### 17. Monitor Model Performance Over Time
**Decision:** Recommend tracking forecast error as new measurements arrive.

**Rationale:**
- The best model today may not be best as data grows.
- Enables automated retraining triggers.
- Provides early warning of degradation.

### 18. Graceful Handling of Missing Features
**Decision:** In the production module, handle missing columns by reindexing and filling with 0.

**Rationale:**
- Future data may lack some engineered features.
- Prevents crashes at prediction time.
- Explicitly documents assumptions (missing = 0).

---

## Tooling & Code Organization

### 19. Modular Production Module (`src/auto_forecast.py`)
**Decision:** Implement all forecasting logic in a single, importable module.

**Rationale:**
- Decouples notebook from production code.
- Enables unit testing and reuse in scripts/APIs.
- Clear API: one function call does everything.

### 20. Use Standard Libraries Only
**Decision:** Rely on pandas, numpy, scikit-learn, and joblib only.

**Rationale:**
- Minimizes dependency risk.
- Widely supported in production environments.
- No heavy ML frameworks required for this problem size.

### 21. Explicit Path Handling with pathlib
**Decision:** Use `pathlib.Path` for all file paths.

**Rationale:**
- Cross-platform compatibility.
- Reduces string concatenation errors.
- Clear intent (file system operations).

---

## Documentation & Reproducibility

### 22. Inline Comments and Docstrings
**Decision:** Provide detailed docstrings for functions and inline comments for non-obvious steps.

**Rationale:**
- Future maintainers (or you) will understand the intent.
- Reduces onboarding time.
- Explains “why” not just “what”.

### 23. Summary Report in Artifacts
**Decision:** Include a metrics summary and comparison table in the saved artifact.

**Rationale:**
- Enables post-hoc analysis without rerunning.
- Supports audit and compliance.
- Facilitates model comparison over time.

### 24. Version Control Friendly Outputs
**Decision:** Save outputs as CSV/JSON/Markdown rather than binary where possible.

**Rationale:**
- Easy to diff and review.
- Human-readable for stakeholders.
- No proprietary formats.

---

## Summary of Key Trade-offs

| Decision | Pro | Con | Mitigation |
|----------|-----|-----|------------|
| Use irregular dataset | Realistic for sparse data | Fewer engineered features | Gap-aware features compensate |
| Include naive baseline | Strong benchmark; simple | May discourage ML use | Still evaluate ML; monitor over time |
| Walk-forward validation | Time-respecting; multiple test points | More complex than random split | Implemented once in module |
| Shallow tree only | Interpretable; low variance | May underfit complex patterns | Add linear models for comparison |
| No hold-out test set | Maximizes training data | No final unbiased estimate | Walk-forward provides realistic estimate |
| Auto-selection module | Production-ready; reduces manual work | Black-box if not documented | Detailed docstrings and artifact contents |

---

## How to Extend This Work

1. **More Data:** As measurements accumulate, revisit model depth and consider ensembles.
2. **Features:** Add external regressors (e.g., system load, team size) if available.
3. **Multi-step Forecast:** Extend to predict 2-3 steps ahead once data supports it.
4. **Probabilistic Forecasting:** Use quantile regression or Bayesian methods for prediction intervals.
5. **Automated Retraining:** Set up a scheduler to re-run the module weekly/monthly.
6. **Explainability Dashboard:** Build a simple UI showing feature contributions and recent errors.

---

## Conclusion

Every design decision balances realism, robustness, and simplicity given the constraints of a tiny, irregular time series. The chosen approach prioritizes:
- Correct temporal modeling (walk-forward, no leakage)
- Production readiness (auto-selection, artifacts, path handling)
- Transparency (baseline, interpretable models, documentation)

This ensures the Capstone project is both academically sound and practically deployable.
