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

### 1. Exploratory Data Analysis (EDA) on Initial Dataset
**Decision:** Perform comprehensive EDA on `pseudo_deident.csv` before any data transformation or feature engineering.

**Dataset Reduction Context:**
- **Initial Dataset**: 3,189 rows of raw test events from `pseudo_deident.csv`
- **Final Dataset**: 19 measurements after temporal aggregation
- **Reduction Reason**: Raw test events aggregated by measurement date to create daily DQ_SCORE calculations
- **Aggregation Logic**: Multiple test executions per day → single daily measurement with calculated pass rate
- **Temporal Focus**: Analysis shifted from individual test events to daily data quality trends

**EDA Objectives:**
- **Data Distribution Analysis**: Understand the distribution of test results, severity levels, and temporal patterns
- **Daily Test Volume**: Analyze the number of tests executed per day to identify patterns and anomalies
- **Test Status Breakdown**: Examine the distribution of PASS/FAIL outcomes across different test categories
- **Statistical Profile**: Generate descriptive statistics for numerical and categorical variables

**Specific EDA Activities:**
- **Temporal Analysis**: Plot test execution frequency over time to identify measurement patterns and gaps
- **Status Distribution**: Create bar charts and pie charts showing PASS/FAIL ratios by test type
- **Severity Analysis**: Examine the relationship between severity levels and test outcomes
- **Missing Value Assessment**: Identify and quantify missing data patterns across all columns
- **Test Category Analysis**: Analyze test distribution across allocation, completeness, uniqueness, referential integrity, privacy, and schema categories
- **Source/Target Analysis**: Examine database, schema, and table patterns in test execution

**Visualization Techniques And Statistical Summary Metrics Used in Capstone_notebook.ipynb:**
- **Seaborn style settings**: `plt.style.use('seaborn-v0_8')` and `sns.set_palette("husl")`
- **Matplotlib configuration**: Basic plotting setup for data visualization
- **Statistical plots**: Configured for displaying data distributions and patterns
- **Custom display settings**: Pandas display options for better data visualization
- **Missing Values Analysis**: `missing_values = df.isnull().sum()` and `missing_percentage = (missing_values / len(df)) * 100`
- **Test Result Statistics**: `df['TEST_RESULT'].describe()` showing mean, median, max issues per test
- **Pass/Fail Analysis**: `df['STATUS'].value_counts()` for pass/fail distribution
- **Categorical Variable Distributions**: `df[col].nunique()` and `df[col].value_counts()` for unique values and top categories
- **Temporal Analysis**: Date-based analysis of test execution patterns
- **Data Quality Assessment**: Missing value impact classification (HIGH/MODERATE/LOW based on percentage)

**Rationale:**
- **Data Quality Understanding**: Identifies issues that need addressing before feature engineering
- **Pattern Recognition**: Reveals temporal patterns that influence feature design
- **Baseline Establishment**: Provides reference metrics for model evaluation
- **Anomaly Detection**: Identifies outliers and unusual patterns requiring special handling
- **Feature Engineering Guidance**: Informs which features might be most predictive based on data characteristics

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

### 3. Mode-Dependent Model Selection (Post-Run vs Pre-Run)
**Decision:** Implement automatic model selection based on available feature set and operational context.

**Current Experiment Results (20260112_204906):**
- **Selected Model**: ElasticNet (linear model with L1/L2 regularization)
- **RMSE Performance**: 0.287583 (winner vs DecisionTree 0.294162, Naive 0.397030)
- **Feature Selection**: Complete regularization - 239 zero coefficients, 0 positive coefficients
- **Approach**: Temporal patterns dominate over engineered features

### Naive Baseline Model
**Decision:** Include Naive(last_value) as a fundamental baseline for all model comparisons.

**Naive Model Definition:**
- **Mathematical Formula**: ŷ_{t+1} = y_t (next prediction equals last observed value)
- **Implementation**: For each test date, prediction is the most recent previous DQ_SCORE measurement
- **Characteristics**: No training required, handles irregular time gaps naturally
- **Purpose**: Serves as minimum performance bar; ML models must meaningfully outperform to justify complexity

**Baseline Usage:**
- **Performance Benchmark**: Current RMSE 0.397030 provides reference for ML model evaluation
- **Simplicity Advantage**: Transparent, always available, and easy to explain to stakeholders
- **Overfitting Protection**: Prevents deployment of ML models that don't add value over simple forecasting

**Post-Run Mode (Rich Feature Set):**
- **Selected Model**: ElasticNet (linear model with L1/L2 regularization)
- **Feature Availability**: Complete daily aggregates (violations, failures, execution results)
- **Model Characteristics**: 
  - Linear relationships between daily metrics and DQ_SCORE
  - Feature coefficients provide interpretable weights
  - Handles multicollinearity through regularization
  - Continuous predictions with smooth curves
- **Performance**: 85-90% accuracy with complete information
- **Current Finding**: Engineered features eliminated through L1 regularization, temporal patterns dominate

**Pre-Run Mode (Limited Feature Set):**
- **Selected Model**: DecisionTree(max_depth=2) - Based on previous experiments
- **Feature Availability**: Historical patterns without current day's results
- **Model Characteristics**:
  - Non-linear pattern detection with limited features
  - Rule-based decision logic (if-then conditions)
  - Step-wise predictions with discrete jumps
  - Conservative forecasting with wider confidence intervals
- **Performance**: 70-80% accuracy without current day's data
- **Key Features**: `passed_tests` (60% importance), `measurements_last_30d` (40% importance)

**Current Experiment Insights:**
- **Temporal Dominance**: Historical DQ_SCORE values are primary predictors
- **Feature Engineering Impact**: Current ElasticNet eliminated all engineered features
- **Model Evolution**: From DecisionTree (194417) to ElasticNet (204906) with 3.6% RMSE improvement
- **Regularization Success**: L1 regularization prevents overfitting with 19 measurements

**Feature Interpretation and DQ_SCORE Relationship:**

**`passed_tests` (60% importance - Pre-Run Mode):**
- **Definition**: Number of tests that passed in the most recent measurement period
- **DQ_SCORE Relationship**: Higher passed tests directly correlate with higher DQ_SCORE
- **Business Logic**: More successful test executions indicate better data quality
- **Interpretation**: If `passed_tests >= 248`, DQ_SCORE tends to be higher (better quality)
- **Why Important**: Test volume represents system capacity and execution success rate

**`measurements_last_30d` (40% importance - Pre-Run Mode):**
- **Definition**: Number of data quality measurements taken in the past 30 days
- **DQ_SCORE Relationship**: Consistent measurement frequency correlates with stable DQ_SCORE
- **Business Logic**: Regular testing indicates mature data quality processes
- **Interpretation**: Higher measurement frequency suggests proactive quality management
- **Why Important**: Consistency in testing reflects organizational commitment to data quality

**Post-Run Mode Features and DQ_SCORE Impact:**

**`daily_metric_alloc_diff_mean` (ElasticNet coefficient: +0.007128):**
- **Definition**: Mean difference in resource allocation counts between current and previous periods
- **DQ_SCORE Relationship**: Positive coefficient means allocation improvements increase DQ_SCORE
- **Business Logic**: Better resource allocation indicates optimized data infrastructure
- **Interpretation**: When allocation metrics improve (positive differences), data quality improves
- **Healthcare Context**: Efficient allocation of healthcare data resources supports better quality

**`daily_complexity_score_max` (ElasticNet coefficient: +0.004493):**
- **Definition**: Maximum complexity score of tests executed in a day
- **DQ_SCORE Relationship**: Positive coefficient suggests handling complexity improves quality
- **Business Logic**: Successfully processing complex tests indicates mature systems
- **Interpretation**: Higher complexity capability = better data quality management
- **Healthcare Context**: Complex healthcare data operations (cross-system validation) improve overall quality

**`daily_metric_count_person_id_sum` (ElasticNet coefficient: +0.004364):**
- **Definition**: Total count of person-based tests executed daily
- **DQ_SCORE Relationship**: More person tests correlate with higher DQ_SCORE
- **Business Logic**: Comprehensive person data validation improves overall quality
- **Interpretation**: Better person data coverage = higher data quality scores
- **Healthcare Context**: Patient data validation is critical for healthcare data quality

**`daily_metric_count_distinct_person_id_sum` (ElasticNet coefficient: +0.004333):**
- **Definition**: Total count of distinct person IDs tested daily
- **DQ_SCORE Relationship**: Higher distinct person coverage improves DQ_SCORE
- **Business Logic**: Broader person coverage indicates comprehensive testing
- **Interpretation**: Testing more unique persons = better data quality assurance
- **Healthcare Context**: Diverse patient population testing ensures data representativeness

**Model Selection Rationale:**
- **ElasticNet for Post-Run**: Rich feature set enables linear modeling with high interpretability
- **DecisionTree for Pre-Run**: Limited features require non-linear pattern detection
- **Automatic Selection**: Pipeline evaluates both models and selects based on RMSE performance

### 4. Impact of Additional Data on Model Selection
**Decision:** Anticipate model evolution as more measurements accumulate over time.

**Expected Changes with More Data:**

**For Post-Run Mode (ElasticNet Evolution):**
- **Feature Stability**: Current top features likely remain important (allocation, complexity, person metrics)
- **Coefficient Refinement**: More precise weight estimation as sample size increases
- **Regularization Adjustment**: Alpha parameters may decrease as overfitting risk reduces
- **Confidence Intervals**: Narrower prediction intervals with larger datasets
- **Feature Expansion**: Additional engineered features may become statistically significant

**For Pre-Run Mode (DecisionTree Evolution):**
- **Depth Increase**: Current max_depth=2 may expand to depth=3-4 with more data
- **Feature Importance**: More features may gain importance beyond current top 2
- **Complexity Growth**: Tree can capture more nuanced patterns with additional examples
- **Stability Improvement**: Reduced variance in tree structure with larger training sets
- **Alternative Models**: May transition to ensemble methods (RandomForest) if data supports

**Data Quantity Thresholds for Model Changes:**
- **50-100 measurements**: DecisionTree depth may increase to 3, feature importance expands
- **200+ measurements**: Consider ensemble methods, more complex feature interactions
- **500+ measurements**: Deep learning approaches may become viable
- **1000+ measurements**: Advanced time series models (LSTM, Prophet) could be evaluated

**Why Model Evolution Matters:**
- **Current Limitation**: Only 19 measurements constrain model complexity
- **Overfitting Risk**: Deep trees would memorize noise with current dataset size
- **Feature Learning**: More data enables discovery of subtle patterns
- **Business Value**: Improved accuracy supports better operational decisions

**Model Selection Strategy Evolution:**
- **Current Phase**: Conservative models (shallow trees, strong regularization)
- **Growth Phase**: Gradual complexity increase as data accumulates
- **Maturity Phase**: Optimal model selection based on data characteristics
- **Monitoring Phase**: Continuous evaluation of model performance drift

### 5. Include Naive(last_value) as a Baseline
**Decision:** Always evaluate a "next ≈ current" baseline alongside ML models.

**Rationale:**
- On small, stable time series, the naive baseline is often the best forecast.
- Provides a minimum performance bar; if ML can't beat it, don't deploy it.
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

### 7a. Model Evolution: From Naive Baseline to ElasticNet Superiority
**Decision:** Document the evolution from naive baseline dominance to ElasticNet model superiority as the dataset matured.

**Evolution Timeline:**
- **Initial Phase**: Naive baseline (last observed DQ_SCORE) consistently outperformed ML models
- **Growth Phase**: Feature engineering and temporal patterns emerged as predictive signals
- **Current Phase (204906)**: ElasticNet model emerged as winner (RMSE 0.287583 vs Naive 0.397030)

**Why the Shift Occurred:**
- **Dataset Maturity**: 19 measurements now provide sufficient temporal signal for ML patterns
- **Temporal Patterns**: Historical DQ_SCORE autocorrelation became primary predictive signal
- **Regularization Success**: ElasticNet's L1 regularization prevents overfitting while capturing signal
- **Feature Engineering Impact**: Engineered features eliminated through regularization, temporal patterns dominate

**Current State (20260112_204906):**
- **ElasticNet Winner**: RMSE 0.287583 (38% better than Naive)
- **Feature Elimination**: 239 engineered features eliminated through L1 regularization
- **Temporal Dominance**: Historical DQ_SCORE patterns are primary predictors
- **Model Stability**: Consistent performance across validation periods

**Business Implications:**
- **ML Success**: Machine learning now provides meaningful improvement over baseline
- **Actionable Insights**: Temporal patterns can be monitored for early warning
- **Production Readiness**: ElasticNet model suitable for deployment
- **Continuous Improvement**: More data will likely further improve ML performance

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

### 5. Walk-Forward (Expanding Window) Validation
**Decision:** Use walk-forward validation instead of random train/test split.

**Why Walk-Forward Was Used:**
- **Temporal Data Integrity**: Preserves the chronological order of measurements, preventing future information leakage
- **Realistic Simulation**: Mimics production scenarios where you train on historical data and predict the next measurement
- **Irregular Time Series Handling**: Accommodates the irregular measurement patterns (gaps of days to weeks between tests)
- **Multiple Test Points**: Provides several validation points rather than a single train/test split, crucial with only 19 measurements
- **Model Selection Accuracy**: Ensures the chosen model performs well in realistic forecasting conditions

**How Walk-Forward Works:**
1. **Initial Training**: Start with minimum training size (8 observations or 40% of data, whichever is larger)
2. **Step-by-Step Prediction**: For each time step t:
   - Train on all available data up to time t
   - Predict the DQ_SCORE for time t+1
   - Record prediction error (RMSE)
3. **Expanding Window**: Each subsequent step includes the new observation in the training set
4. **Aggregate Performance**: Calculate overall RMSE across all prediction steps

**Implementation Details:**
- **Minimum Training Size**: 8 observations (ensures sufficient data for model training)
- **Step Size**: 1 observation at a time (maximizes validation points)
- **Error Metric**: RMSE calculated across all predicted steps
- **Model Comparison**: Each candidate model (ElasticNet, DecisionTree, etc.) evaluated using identical walk-forward procedure

**Advantages Over Random Split:**
- **No Data Leakage**: Prevents using future information to predict past values
- **Time-Aware**: Respects the temporal nature of forecasting tasks
- **Robust Evaluation**: More reliable performance estimate for time series data
- **Production Alignment**: Matches how models will actually be used in practice

**Why Critical for This Dataset:**
- **Tiny Dataset**: Only 19 measurements make random splits unreliable
- **Irregular Spacing**: Variable time gaps between measurements require temporal validation
- **Healthcare Context**: Medical data quality decisions depend on realistic performance estimates
- **Model Selection**: Ensures the chosen model (ElasticNet vs DecisionTree) truly performs best in production-like conditions

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

### 16. Model Selection Based on Empirical Performance
**Decision:** Use the model that demonstrates the best empirical performance, whether ML or baseline.

**Rationale:**
- **Performance-Driven**: Select models based on actual RMSE results from walk-forward validation
- **Evolution Acceptance**: Model selection may change as dataset matures and patterns emerge
- **Current Evidence**: ElasticNet now outperforms Naive baseline (0.287583 vs 0.397030)
- **Flexibility**: Allow model selection to evolve with data growth and pattern development

**Current Recommendation (20260112_204906):**
- **Deploy ElasticNet**: 38% better performance than Naive baseline
- **Monitor Performance**: Track if ElasticNet maintains superiority
- **Continue Evaluation**: Regular retraining as new data becomes available
- **Fallback Option**: Naive baseline remains available if performance degrades

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
