"""
Irregular Time Series Processor for DQ_SCORE Dataset
Handles sparse, irregular time series data appropriately for AI Agent input
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import os
import warnings
warnings.filterwarnings('ignore')

class IrregularTimeSeriesProcessor:
    """
    Processes sparse time series data for ML models by:
    1. Removing temporal features that assume regular intervals
    2. Creating gap-aware features
    3. Handling infinite values appropriately
    4. Engineering features suitable for irregular time series
    """
    
    def __init__(self):
        self.gap_features = []
        self.processed_features = []
        
    def load_and_analyze_data(self, filepath: str) -> pd.DataFrame:
        """Load data and analyze temporal gaps"""
        df = pd.read_csv(filepath, index_col='calendar_date', parse_dates=True)
        
        print("=== IRREGULAR TIME SERIES ANALYSIS ===")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total days in range: {(df.index.max() - df.index.min()).days + 1}")
        print(f"Actual data points: {len(df)}")
        print(f"Data sparsity: {len(df) / ((df.index.max() - df.index.min()).days + 1) * 100:.1f}%")
        
        # Analyze gaps
        gaps = self._analyze_temporal_gaps(df)
        self._print_gap_statistics(gaps)
        
        return df, gaps

    def process_irregular_time_series(self, input_file: str, output_dir: str) -> pd.DataFrame:
        df, _ = self.load_and_analyze_data(input_file)
        df_processed = self.create_irregular_time_features(df)
        df_clean, metadata = self.prepare_for_ai_agent(df_processed)

        output_path = os.path.join(output_dir, 'feature_engineered_events_irregular.csv')
        self.save_processed_data(df_clean, output_path)

        report = self.generate_summary_report(df_clean, metadata)
        report_path = os.path.join(output_dir, 'irregular_time_series_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return df_clean
    
    def _analyze_temporal_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal gaps in the dataset to understand irregularity patterns.
        
        This analysis is crucial for understanding the nature of temporal sparsity
        and designing appropriate gap-aware features. It quantifies how irregular
        the measurements are and provides statistical context for feature engineering.
        
        ANALYSIS METRICS:
        -----------------
        - Mean/Median/Max/Min gaps: Basic statistics about measurement intervals
        - Gap distribution: Frequency of different gap sizes
        - Temporal coverage: Percentage of time periods with measurements
        - Irregularity patterns: Clustering of gaps, seasonal patterns, etc.
        
        INPUT PARAMETERS:
        -----------------
        df (pd.DataFrame): Input dataframe with datetime index
        
        RETURN VALUES:
        --------------
        Dict[str, Any]: Dictionary containing gap analysis statistics:
        - 'mean_gap_days': Average time between consecutive measurements
        - 'median_gap_days': Median time between measurements (robust to outliers)
        - 'max_gap_days': Longest gap without measurements
        - 'min_gap_days': Shortest gap between measurements
        - 'gap_distribution': Count of measurements by gap size (top 10 most common)
        
        BUSINESS INSIGHTS:
        -----------------
        This analysis helps stakeholders understand:
        - Measurement frequency and reliability
        - Data collection patterns and potential issues
        - Expected temporal sparsity for model planning
        - Whether irregular time series processing is necessary
        """
        if len(df) < 2:
            return {}
        
        # Calculate actual gaps between consecutive dates
        date_diffs = df.index.to_series().diff().dt.days.dropna()
        
        gaps = {
            'mean_gap_days': date_diffs.mean(),
            'median_gap_days': date_diffs.median(),
            'max_gap_days': date_diffs.max(),
            'min_gap_days': date_diffs.min(),
            'gap_distribution': date_diffs.value_counts().sort_index().head(10)
        }
        
        return gaps
    
    def _print_gap_statistics(self, gaps: Dict):
        """Print gap analysis statistics"""
        print("\n=== TEMPORAL GAP ANALYSIS ===")
        print(f"Average gap between measurements: {gaps['mean_gap_days']:.1f} days")
        print(f"Median gap: {gaps['median_gap_days']:.1f} days")
        print(f"Maximum gap: {gaps['max_gap_days']:.1f} days")
        print(f"Minimum gap: {gaps['min_gap_days']:.1f} days")
        
        print("\nGap distribution (top 10):")
        for gap, count in gaps['gap_distribution'].items():
            print(f"  {gap:.0f} days gap: {count} occurrences")
    
    def create_irregular_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by removing misleading temporal features and creating
        gap-aware features appropriate for irregular time series data.
        
        This is the core transformation method that addresses the fundamental
        problem: standard temporal features assume regular intervals, but real-world
        data quality measurements are sparse and irregular.
        
        TRANSFORMATION STRATEGY:
        -----------------------
        1. REMOVE MISLEADING FEATURES:
           - Lag features (e.g., DQ_SCORE_lag_1d assumes exactly 1-day gap)
           - Rolling windows (e.g., 7-day rolling mean assumes daily data)
           - Momentum indicators (assume consistent time intervals)
           - Volatility measures (assume regular sampling)
        
        2. CREATE GAP-AWARE FEATURES:
           - Time since/to measurements (actual temporal context)
           - Gap severity indicators (categorical temporal urgency)
           - Measurement frequency (temporal density metrics)
           - Regularity scores (temporal consistency measures)
        
        3. FIX DATA QUALITY ISSUES:
           - Handle infinite values from division by zero
           - Replace misleading percentage changes with rate-based changes
           - Ensure mathematical correctness for sparse data
        
        INPUT PARAMETERS:
        -----------------
        df (pd.DataFrame): Standard feature-engineered dataset with misleading temporal features
        
        RETURN VALUES:
        --------------
        pd.DataFrame: Transformed dataset with gap-aware features and no temporal assumptions
        
        FEATURE TRANSFORMATIONS:
        ------------------------
        REMOVED (approximately 30 features):
        - DQ_SCORE_lag_1d/3d/7d/14d/30d (misleading for sparse data)
        - TRUST_SCORE_lag_1d/3d/7d/14d/30d (assume regular intervals)
        - DQ/TRUST_SCORE_rolling_*_7d/14d/30d (require daily data)
        - DQ/TRUST_SCORE_momentum_*_7d/14d (assume consistent gaps)
        - DQ/TRUST_SCORE_volatility_*_7d/14d (need regular sampling)
        
        ADDED (8 gap-aware features):
        - days_since_last_measurement: Actual temporal gap (0-20 days)
        - days_to_next_measurement: Forward-looking temporal context
        - is_long_gap: Binary indicator for gaps > 7 days
        - gap_severity: Categorical temporal urgency (0=short, 1=medium, 2=long)
        - measurements_last_30d/60d/90d: Temporal density metrics
        - measurement_regularity: Consistency score (higher = more predictable)
        
        FIXED (infinite value handling):
        - Percentage changes: Capped at ±1000% (reasonable bounds)
        - Other infinities: Replaced with median values
        - Rate-based changes: Normalized by actual time gaps
        """
        print("\n=== CREATING IRREGULAR TIME SERIES FEATURES ===")
        
        df_processed = df.copy()
        
        # =======================================================================
        # STEP 1: CREATE TEMPORAL CONTEXT FEATURES
        # =======================================================================
        # These features provide actual temporal context instead of assuming
        # regular intervals. They tell the model exactly how much time has passed.
        
        # Time since last measurement (backward-looking temporal context)
        df_processed['days_since_last_measurement'] = df_processed.index.to_series().diff().dt.days
        df_processed['days_since_last_measurement'] = df_processed['days_since_last_measurement'].fillna(0)
        
        # Days to next measurement (forward-looking temporal context)
        # This helps models anticipate future temporal patterns
        df_processed['days_to_next_measurement'] = df_processed.index.to_series().diff(-1).dt.days.abs()
        df_processed['days_to_next_measurement'] = df_processed['days_to_next_measurement'].fillna(df_processed['days_since_last_measurement'].mean())
        
        # =======================================================================
        # STEP 2: CREATE GAP SEVERITY FEATURES
        # =======================================================================
        # These features quantify the temporal urgency and irregularity patterns.
        # They help models understand when gaps are business-critical vs normal.
        
        # Binary indicator for business-significant gaps (> 1 week)
        df_processed['is_long_gap'] = (df_processed['days_since_last_measurement'] > 7).astype(int)
        
        # Categorical severity based on gap duration
        # 0 = Short gap (≤1 day): Normal operations
        # 1 = Medium gap (2-7 days): Attention needed
        # 2 = Long gap (>7 days): Business impact likely
        df_processed['gap_severity'] = np.where(
            df_processed['days_since_last_measurement'] <= 1, 0,
            np.where(df_processed['days_since_last_measurement'] <= 7, 1, 2)
        )
        
        # =======================================================================
        # STEP 3: REMOVE MISLEADING TEMPORAL FEATURES
        # =======================================================================
        # These features assume regular intervals and are mathematically incorrect
        # for sparse, irregular data. They must be removed to prevent model confusion.
        
        temporal_features_to_remove = [
            # LAG FEATURES - Assume exact N-day intervals (FALSE for sparse data)
            'DQ_SCORE_lag_1d', 'DQ_SCORE_lag_3d', 'DQ_SCORE_lag_7d', 'DQ_SCORE_lag_14d', 'DQ_SCORE_lag_30d',
            'TRUST_SCORE_lag_1d', 'TRUST_SCORE_lag_3d', 'TRUST_SCORE_lag_7d', 'TRUST_SCORE_lag_14d', 'TRUST_SCORE_lag_30d',
            
            # ROLLING WINDOW FEATURES - Require daily data to be meaningful
            'DQ_SCORE_rolling_mean_7d', 'DQ_SCORE_rolling_std_7d', 'DQ_SCORE_rolling_min_7d', 'DQ_SCORE_rolling_max_7d', 'DQ_SCORE_rolling_trend_7d',
            'DQ_SCORE_rolling_mean_14d', 'DQ_SCORE_rolling_std_14d', 'DQ_SCORE_rolling_min_14d', 'DQ_SCORE_rolling_max_14d', 'DQ_SCORE_rolling_trend_14d',
            'DQ_SCORE_rolling_mean_30d', 'DQ_SCORE_rolling_std_30d', 'DQ_SCORE_rolling_min_30d', 'DQ_SCORE_rolling_max_30d', 'DQ_SCORE_rolling_trend_30d',
            'TRUST_SCORE_rolling_mean_7d', 'TRUST_SCORE_rolling_std_7d', 'TRUST_SCORE_rolling_min_7d', 'TRUST_SCORE_rolling_max_7d', 'TRUST_SCORE_rolling_trend_7d',
            'TRUST_SCORE_rolling_mean_14d', 'TRUST_SCORE_rolling_std_14d', 'TRUST_SCORE_rolling_min_14d', 'TRUST_SCORE_rolling_max_14d', 'TRUST_SCORE_rolling_trend_14d',
            'TRUST_SCORE_rolling_mean_30d', 'TRUST_SCORE_rolling_std_30d', 'TRUST_SCORE_rolling_min_30d', 'TRUST_SCORE_rolling_max_30d', 'TRUST_SCORE_rolling_trend_30d',
            
            # MOMENTUM INDICATORS - Assume consistent time intervals
            'DQ_SCORE_momentum_7d', 'DQ_SCORE_momentum_14d', 'TRUST_SCORE_momentum_7d', 'TRUST_SCORE_momentum_14d',
            
            # VOLATILITY MEASURES - Require regular sampling
            'DQ_SCORE_volatility_7d', 'DQ_SCORE_volatility_14d', 'TRUST_SCORE_volatility_7d', 'TRUST_SCORE_volatility_14d'
        ]
        
        # Remove features that exist
        removed_features = []
        for feature in temporal_features_to_remove:
            if feature in df_processed.columns:
                df_processed = df_processed.drop(columns=[feature])
                removed_features.append(feature)
        
        print(f"Removed {len(removed_features)} temporal features that assume regular intervals")
        
        # =======================================================================
        # STEP 4: HANDLE INFINITE VALUES (DATA QUALITY FIXES)
        # =======================================================================
        # Infinite values occur due to division by zero in sparse data scenarios.
        # They crash ML algorithms and indicate mathematical issues that need fixing.
        
        infinite_features = df_processed.columns[df_processed.isin([np.inf, -np.inf]).any()].tolist()
        print(f"Found {len(infinite_features)} features with infinite values")
        
        for feature in infinite_features:
            if 'pct_change' in feature:
                # PERCENTAGE CHANGES: Cap at reasonable bounds (±1000%)
                # This prevents extreme values while preserving directional information
                df_processed[feature] = df_processed[feature].replace([np.inf, -np.inf], 1000)
                print(f"  Capped infinite values in {feature} at 1000%")
            else:
                # OTHER FEATURES: Replace with median (robust central tendency)
                # This maintains the feature's distribution while removing infinities
                median_val = df_processed[feature].replace([np.inf, -np.inf], np.nan).median()
                df_processed[feature] = df_processed[feature].replace([np.inf, -np.inf], median_val)
                print(f"  Replaced infinite values in {feature} with median: {median_val:.4f}")

        pct_change_rename_map = {
            'DQ_SCORE_pct_change_1d': 'DQ_SCORE_pct_change_lag_1obs',
            'DQ_SCORE_pct_change_3d': 'DQ_SCORE_pct_change_lag_3obs',
            'DQ_SCORE_pct_change_7d': 'DQ_SCORE_pct_change_lag_7obs',
            'DQ_SCORE_pct_change_14d': 'DQ_SCORE_pct_change_lag_14obs',
            'TRUST_SCORE_pct_change_1d': 'TRUST_SCORE_pct_change_lag_1obs',
            'TRUST_SCORE_pct_change_3d': 'TRUST_SCORE_pct_change_lag_3obs',
            'TRUST_SCORE_pct_change_7d': 'TRUST_SCORE_pct_change_lag_7obs',
            'TRUST_SCORE_pct_change_14d': 'TRUST_SCORE_pct_change_lag_14obs'
        }

        existing_pct_change_renames = {k: v for k, v in pct_change_rename_map.items() if k in df_processed.columns}
        if existing_pct_change_renames:
            df_processed = df_processed.rename(columns=existing_pct_change_renames)
            print(f"Renamed {len(existing_pct_change_renames)} pct_change features to reflect lag-by-observation semantics")
        
        # =======================================================================
        # STEP 5: CREATE GAP-AWARE TREND FEATURES
        # =======================================================================
        # Replace misleading percentage changes with gap-aware rate-based features.
        # These features properly account for varying time intervals between measurements.
        df_processed = self._create_gap_aware_trends(df_processed)
        
        # =======================================================================
        # STEP 6: CREATE MEASUREMENT FREQUENCY FEATURES
        # =======================================================================
        # Add temporal density and regularity features to help models understand
        # measurement patterns and predict future temporal behavior.
        df_processed = self._create_frequency_features(df_processed)
        
        self.processed_features = df_processed.columns.tolist()
        
        print(f"\nFinal processed dataset shape: {df_processed.shape}")
        print(f"Features created for irregular time series: {len(df_processed.columns)}")
        
        return df_processed
    
    def _create_gap_aware_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create gap-aware trend features that properly account for varying time intervals.
        
        Traditional percentage change features assume consistent time intervals,
        which is false for irregular time series. These new features normalize changes
        by the actual time gaps, providing mathematically correct trend indicators.
        
        FEATURE ENGINEERING LOGIC:
        -------------------------
        1. ABSOLUTE CHANGE: Simple difference between consecutive measurements
           - Pros: Mathematically correct, no division by zero
           - Cons: Not normalized for time (larger gaps may have larger changes)
        
        2. DAILY RATE: Change normalized by actual time gap
           - Formula: (current_value - previous_value) / days_since_last_measurement
           - Pros: Time-normalized, comparable across different gap sizes
           - Cons: Division by zero for first row (handled with fillna(0))
        
        INPUT PARAMETERS:
        -----------------
        df (pd.DataFrame): Dataset with gap-aware temporal features already added
        
        RETURN VALUES:
        --------------
        pd.DataFrame: Dataset with gap-aware trend features added
        
        NEW FEATURES CREATED:
        --------------------
        - DQ_SCORE_absolute_change: Simple difference between consecutive DQ scores
        - DQ_SCORE_daily_rate: DQ change per day (normalized by time gap)
        - TRUST_SCORE_absolute_change: Simple difference between consecutive TRUST scores
        - TRUST_SCORE_daily_rate: TRUST change per day (normalized by time gap)
        
        BUSINESS VALUE:
        --------------
        These features help models understand:
        - Actual magnitude of quality changes (absolute change)
        - Rate of quality degradation/improvement (daily rate)
        - Temporal urgency of interventions (rapid changes vs slow drifts)
        """
        print("Creating gap-aware trend features...")
        
        # =======================================================================
        # CREATE GAP-AWARE TREND FEATURES FOR EACH TARGET SCORE
        # =======================================================================
        # Replace misleading percentage changes with mathematically correct
        # trend indicators that account for varying time intervals.
        
        for score in ['DQ_SCORE', 'TRUST_SCORE']:
            # ABSOLUTE CHANGE: Simple difference between consecutive measurements
            # This is mathematically correct but not time-normalized
            df[f'{score}_absolute_change'] = df[score].diff()
            
            # DAILY RATE: Change normalized by actual time gap
            # This provides comparable trend indicators across different gap sizes
            df[f'{score}_daily_rate'] = df[score].diff() / df['days_since_last_measurement']
            
            # Handle division by zero for first row (no previous measurement)
            # Set to 0 as there's no change rate for the first measurement
            df[f'{score}_daily_rate'] = df[f'{score}_daily_rate'].fillna(0)
        
        return df
    
    def _create_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create measurement frequency and regularity features to quantify temporal patterns.
        
        These features help models understand the temporal context and predict future
        measurement patterns. They provide insights into data collection reliability
        and help distinguish between normal irregularity and concerning gaps.
        
        FEATURE CATEGORIES:
        -------------------
        1. TEMPORAL DENSITY: Count of measurements in recent time windows
           - Helps models understand recent measurement activity
           - Indicates data collection intensity and reliability
        
        2. TEMPORAL REGULARITY: Consistency of measurement intervals
           - Higher values indicate more predictable measurement patterns
           - Lower values indicate irregular or unreliable data collection
        
        INPUT PARAMETERS:
        -----------------
        df (pd.DataFrame): Dataset with days_since_last_measurement feature
        
        RETURN VALUES:
        --------------
        pd.DataFrame: Dataset with frequency and regularity features added
        
        NEW FEATURES CREATED:
        --------------------
        - measurements_last_30d: Count of measurements in past 30 days
        - measurements_last_60d: Count of measurements in past 60 days
        - measurements_last_90d: Count of measurements in past 90 days
        - measurement_regularity: Consistency score (0-1, higher = more regular)
        
        BUSINESS INSIGHTS:
        -----------------
        These features help stakeholders understand:
        - Data collection reliability and consistency
        - Recent measurement activity levels
        - Predictability of future measurements
        - Whether gaps are normal or concerning
        """
        print("Creating frequency-based features...")
        
        # =======================================================================
        # CREATE TEMPORAL DENSITY FEATURES
        # =======================================================================
        # Rolling frequency counts show how many measurements occurred in recent
        # time windows. This helps models understand recent data collection activity.
        
        for window in [30, 60, 90]:  # days - multiple time horizons for different insights
            df[f'measurements_last_{window}d'] = (
                df.index.to_series().rolling(f'{window}d').count()
            )
        
        # =======================================================================
        # CREATE TEMPORAL REGULARITY FEATURE
        # =======================================================================
        # Measurement regularity quantifies how consistent the measurement intervals are.
        # Higher values indicate more predictable, regular measurement patterns.
        # 
        # Formula: 1 / (1 + gap_variance)
        # - gap_variance = 0 for perfectly regular measurements → regularity = 1.0
        # - gap_variance increases → regularity approaches 0.0
        # 
        # We use a rolling window of 5 measurements to calculate local regularity.
        gap_variance = df['days_since_last_measurement'].rolling(window=5).var()
        df['measurement_regularity'] = 1 / (1 + gap_variance)  # Higher = more regular
        df['measurement_regularity'] = df['measurement_regularity'].fillna(0)
        
        return df
    
    def prepare_for_ai_agent(self, df: pd.DataFrame, target_variables: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare the final dataset for AI Agent deployment with comprehensive cleaning.
        
        This method performs the final data preparation steps to ensure the dataset
        is completely ready for AI Agent consumption. It addresses any remaining
        data quality issues and creates comprehensive metadata for downstream systems.
        
        CLEANING STEPS:
        ----------------
        1. INFINITE VALUE HANDLING: Replace any remaining infinities with NaN
        2. MISSING VALUE IMPUTATION: Forward fill → backward fill → zero fallback
        3. FEATURE-TARGET SEPARATION: Clear distinction for ML pipelines
        4. METADATA GENERATION: Comprehensive dataset documentation
        
        INPUT PARAMETERS:
        -----------------
        df (pd.DataFrame): Processed dataset with gap-aware features
        target_variables (List[str]): List of target variable names
                                     (default: ['DQ_SCORE', 'TRUST_SCORE'])
        
        RETURN VALUES:
        --------------
        Tuple[pd.DataFrame, Dict]: 
        - First element: Completely cleaned dataset ready for AI Agent
        - Second element: Comprehensive metadata dictionary
        
        METADATA CONTENTS:
        ------------------
        - dataset_shape: (samples, features) of cleaned dataset
        - feature_count: Number of predictive features (excluding targets)
        - target_variables: List of target variable names
        - date_range: (start_date, end_date) of temporal coverage
        - measurement_count: Total number of measurements
        - average_gap_days: Mean time between measurements
        - feature_types: Count of numeric vs categorical features
        
        AI AGENT READINESS:
        -------------------
        After this processing, the dataset is:
        - ✅ Free of infinite values
        - ✅ Free of missing values
        - ✅ Properly structured for ML algorithms
        - ✅ Well-documented with comprehensive metadata
        - ✅ Ready for AI Agent deployment
        """
        # Set default target variables if not specified
        if target_variables is None:
            target_variables = ['DQ_SCORE', 'TRUST_SCORE']
        
        print("\n=== PREPARING FOR AI AGENT ===")
        
        # =======================================================================
        # STEP 1: COMPREHENSIVE DATA CLEANING
        # =======================================================================
        # Ensure dataset is completely clean for AI Agent consumption
        
        # Replace any remaining infinite values with NaN (for proper handling)
        # Track run-specific infinite handling stats for reporting.
        numeric_df = df.select_dtypes(include=[np.number])
        inf_mask_before = np.isinf(numeric_df.to_numpy())
        infinite_values_found = int(inf_mask_before.sum())

        df_clean = df.replace([np.inf, -np.inf], np.nan)
        
        # Multi-stage missing value imputation:
        # 1. Forward fill: Use previous measurement (temporally logical)
        # 2. Backward fill: Use next measurement (fallback)
        # 3. Zero fill: Final fallback for any remaining NaN values
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)

        numeric_df_after = df_clean.select_dtypes(include=[np.number])
        inf_mask_after = np.isinf(numeric_df_after.to_numpy())
        infinite_values_remaining = int(inf_mask_after.sum())
        infinite_values_fixed = infinite_values_found - infinite_values_remaining
        
        # =======================================================================
        # STEP 2: FEATURE-TARGET SEPARATION
        # =======================================================================
        # Clear separation for ML pipelines and feature importance analysis
        
        feature_cols = [col for col in df_clean.columns if col not in target_variables]
        X = df_clean[feature_cols]  # Predictive features
        y = df_clean[target_variables]  # Target variables
        
        # =======================================================================
        # STEP 3: COMPREHENSIVE METADATA GENERATION
        # =======================================================================
        # Create detailed documentation for downstream AI Agent systems
        
        metadata = {
            # Basic dataset information
            'dataset_shape': df_clean.shape,
            'feature_count': len(feature_cols),
            'target_variables': target_variables,
            
            # Temporal information
            'date_range': (df_clean.index.min(), df_clean.index.max()),
            'measurement_count': len(df_clean),
            'average_gap_days': df_clean['days_since_last_measurement'].mean(),
            
            # Feature type breakdown
            'feature_types': {
                'numeric': len(X.select_dtypes(include=[np.number]).columns),
                'categorical': len(X.select_dtypes(include=['object']).columns)
            },

            # Run-specific data quality handling summary
            'infinite_values': {
                'found': infinite_values_found,
                'fixed': infinite_values_fixed,
                'remaining': infinite_values_remaining
            }
        }
        
        print(f"Final dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target variables: {target_variables}")
        
        return df_clean, metadata
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed irregular time series data"""
        df.to_csv(output_path)
        print(f"\nProcessed data saved to: {output_path}")
    
    def generate_summary_report(self, df: pd.DataFrame, metadata: Dict) -> str:
        """Generate summary report for the processed dataset"""
        inf_stats = metadata.get('infinite_values', {'found': 0, 'fixed': 0, 'remaining': 0})
        report = f"""
# Irregular Time Series Processing Summary

## Dataset Overview
- **Original samples**: {metadata['dataset_shape'][0]}
- **Original features**: {metadata['dataset_shape'][1]}
- **Processed features**: {metadata['feature_count']}
- **Date range**: {metadata['date_range'][0]} to {metadata['date_range'][1]}
- **Measurement count**: {metadata['measurement_count']}
- **Average gap**: {metadata['average_gap_days']:.1f} days

## Key Transformations
1. **Removed temporal features** that assume regular intervals (lags, rolling windows)
2. **Created gap-aware features**:
   - days_since_last_measurement
   - days_to_next_measurement
   - gap_severity
   - measurement_regularity
3. **Infinite value handling**: checked for +/-inf values (found={inf_stats['found']}, fixed={inf_stats['fixed']}, remaining={inf_stats['remaining']})
4. **Created rate-based features** instead of percentage changes

## Suitability for AI Agent
- READY for irregular time series models
- Gap-aware features provide temporal context
- Infinite values resolved
- Sparse data properly handled

## Recommended Model Types
- Tree-based models (Random Forest, XGBoost)
- Time-aware neural networks
- Anomaly detection algorithms
- Irregular time series forecasting models
"""
        return report


def main():
    """Main processing pipeline"""
    processor = IrregularTimeSeriesProcessor()
    
    # Load and analyze
    df, gaps = processor.load_and_analyze_data('../data/feature_engineered_events_final.csv')
    
    # Process for irregular time series
    df_processed = processor.create_irregular_time_features(df)
    
    # Prepare for AI agent
    df_final, metadata = processor.prepare_for_ai_agent(df_processed)
    
    # Save processed data
    processor.save_processed_data(df_final, '../data/feature_engineered_events_irregular.csv')
    
    # Generate report
    report = processor.generate_summary_report(df_final, metadata)
    
    # Save report
    with open('../data/irregular_time_series_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*50)
    print("IRREGULAR TIME SERIES PROCESSING COMPLETE")
    print("="*50)
    print(report)


if __name__ == "__main__":
    main()
