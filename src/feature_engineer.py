"""
Healthcare Data Quality Feature Engineering Pipeline

This module provides comprehensive feature engineering for healthcare data quality events,
including JSON parsing, status extraction, missing value handling, and DQ_SCORE calculation.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import ast

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareDQFeatureEngineer:
    """
    Comprehensive feature engineering for healthcare data quality events.
    
    This class handles:
    - JSON parsing from DQ_EVENT column
    - Status extraction from test_result
    - Missing value handling for optional fields
    - DQ_SCORE and TRUST_SCORE calculation
    - Healthcare-specific categorization
    - Feature engineering for regression modeling
    """
    
    def __init__(self):
        self.severity_weights = {'HIGH SEVERITY': 3, 'MEDIUM SEVERITY': 2, 'LOW SEVERITY': 1}
        self.NOT_APPLICABLE = "NOT_APPLICABLE"
        self.NO_VIOLATIONS = "NO_VIOLATIONS"
    
    def parse_dq_event_with_status(self, dq_event: Dict[str, Any]) -> Dict[str, Any]:
        """Parse DQ_EVENT dictionary extracting status and other features"""
        features = {}
        
        # Extract status from top level (primary source for DQ_SCORE)
        features['status'] = dq_event.get('status', self.NOT_APPLICABLE)
        
        # Create status indicators
        features['test_passed'] = 1 if features['status'] == 'pass' else 0
        features['test_failed'] = 1 if features['status'] == 'fail' else 0
        features['test_error'] = 1 if features['status'] == 'error' else 0
        features['test_skipped'] = 1 if features['status'] == 'skipped' else 0
        
        # Extract test info
        features['test_family'] = dq_event.get('test_family', self.NOT_APPLICABLE)
        features['test_category'] = dq_event.get('test_category', self.NOT_APPLICABLE)
        
        # Extract metrics if available
        if 'metrics' in dq_event:
            metrics = dq_event['metrics']
            features['execution_time'] = 0  # Not available in current data structure
            features['records_processed'] = metrics.get('curr_alloc_count', 0)
            features['records_failed'] = 0  # Not available in current data structure
        else:
            features['execution_time'] = 0
            features['records_processed'] = 0
            features['records_failed'] = 0
        
        # For allocation tests, no violations means passed status
        features['has_violations'] = 0 if features['status'] == 'pass' else 1
        features['violation_count'] = 0 if features['status'] == 'pass' else 1
        # Use original severity from data, don't hardcode!
        if dq_event.get('original_severity'):
            features['violation_severity'] = dq_event['original_severity']
        else:
            features['violation_severity'] = self.NO_VIOLATIONS if features['status'] == 'pass' else 'HIGH SEVERITY'
        
        # Extract additional metrics
        if 'metrics' in dq_event:
            metrics = dq_event['metrics']
            for key, value in metrics.items():
                features[f'metric_{key}'] = value
        
        # Extract object info with missing value handling
        if 'objects' in dq_event:
            objects = dq_event['objects']
            
            # Source object information
            if 'source' in objects:
                source = objects['source']
                features['source_database'] = source.get('database', self.NOT_APPLICABLE)
                features['source_table'] = source.get('table', self.NOT_APPLICABLE)
                # Handle nan values for column
                column_value = source.get('column')
                features['source_column'] = column_value if pd.notna(column_value) else self.NOT_APPLICABLE
                features['source_object_level'] = source.get('object_level', self.NOT_APPLICABLE)
            else:
                for suffix in ['database', 'table', 'column', 'object_level']:
                    features[f'source_{suffix}'] = self.NOT_APPLICABLE
            
            # Target object information
            if 'target' in objects:
                target = objects['target']
                features['target_database'] = target.get('database', self.NOT_APPLICABLE)
                features['target_table'] = target.get('table', self.NOT_APPLICABLE)
                # Handle nan values for column
                column_value = target.get('column')
                features['target_column'] = column_value if pd.notna(column_value) else self.NOT_APPLICABLE
                features['target_object_level'] = target.get('object_level', self.NOT_APPLICABLE)
            else:
                for suffix in ['database', 'table', 'column', 'object_level']:
                    features[f'target_{suffix}'] = self.NOT_APPLICABLE
        else:
            for prefix in ['source', 'target']:
                for suffix in ['database', 'table', 'column', 'object_level']:
                    features[f'{prefix}_{suffix}'] = self.NOT_APPLICABLE
        
        return features
    
    def extract_json_features_with_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from DQ_EVENT JSON using status from test_result"""
        json_features = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                # Preprocess the string to replace nan with None for ast.literal_eval
                dq_event_str = row['DQ_EVENT'].replace('nan', 'None')
                # Use ast.literal_eval to parse Python dictionary strings with single quotes
                dq_event = ast.literal_eval(dq_event_str)
                
                # Add original severity from DataFrame
                original_severity = row.get('SEVERITY_LEVEL', 'NO_VIOLATIONS')
                dq_event['original_severity'] = original_severity
                
                features = self.parse_dq_event_with_status(dq_event)
                features['date'] = row.name
                json_features.append(features)
                
                if (idx + 1) % 1000 == 0:
                    logger.info(f"Processed {idx + 1} records")
                    
            except (ValueError, SyntaxError) as e:
                # Handle parsing errors silently - count them but don't spam logs
                if not hasattr(self, '_json_error_count'):
                    self._json_error_count = 0
                self._json_error_count += 1
                
                # Only log the first few errors for debugging
                if self._json_error_count <= 3:
                    logger.warning(f"DQ_EVENT parsing error at index {row.name}: {str(e)[:100]}...")
                elif self._json_error_count == 4:
                    logger.warning(f"Additional DQ_EVENT parsing errors detected (suppressing further logs)")
                
                error_features = {
                    'date': row.name,
                    'status': 'ERROR',
                    'test_passed': 0,
                    'test_failed': 0,
                    'test_error': 1,
                    'test_skipped': 0,
                    'test_family': self.NOT_APPLICABLE,
                    'test_category': self.NOT_APPLICABLE,
                    'execution_time': 0,
                    'records_processed': 0,
                    'records_failed': 0,
                    'has_violations': 0,
                    'violation_count': 0,
                    'violation_severity': 'UNKNOWN',
                    'source_database': self.NOT_APPLICABLE,
                    'source_table': self.NOT_APPLICABLE,
                    'source_column': self.NOT_APPLICABLE,
                    'source_object_level': self.NOT_APPLICABLE,
                    'target_database': self.NOT_APPLICABLE,
                    'target_table': self.NOT_APPLICABLE,
                    'target_column': self.NOT_APPLICABLE,
                    'target_object_level': self.NOT_APPLICABLE
                }
                json_features.append(error_features)
        
        # Log summary of parsing
        if hasattr(self, '_json_error_count') and self._json_error_count > 0:
            logger.warning(f"Total DQ_EVENT parsing errors: {self._json_error_count} out of {len(df)} records")
        
        json_df = pd.DataFrame(json_features)
        return json_df.set_index('date')
    
    def create_status_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on status patterns"""
        
        # Status indicators
        df['is_passed'] = (df['status'] == 'pass').astype(int)
        df['is_failed'] = (df['status'] == 'fail').astype(int)
        df['is_error'] = (df['status'] == 'error').astype(int)
        df['is_skipped'] = (df['status'] == 'skipped').astype(int)
        
        # Status by test family
        for test_family in df['test_family'].unique():
            if test_family != self.NOT_APPLICABLE:
                family_mask = df['test_family'] == test_family
                df[f'{test_family}_passed'] = (family_mask & df['is_passed']).astype(int)
                df[f'{test_family}_failed'] = (family_mask & df['is_failed']).astype(int)
                df[f'{test_family}_error'] = (family_mask & df['is_error']).astype(int)
        
        # Status consistency checks
        df['status_consistent_with_results'] = (
            ((df['status'] == 'pass') & (df['has_violations'] == 0)) |
            ((df['status'] == 'fail') & (df['has_violations'] == 1)) |
            (df['status'].isin(['error', 'skipped']))
        ).astype(int)
        
        return df
    
    def create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary indicators for missing optional fields"""
        optional_fields = [
            'source_database', 'source_table', 'source_column', 'source_object_level',
            'target_database', 'target_table', 'target_column', 'target_object_level'
        ]
        
        for field in optional_fields:
            # Create indicator: 1 if present, 0 if NOT_APPLICABLE
            df[f'has_{field}'] = (df[field] != self.NOT_APPLICABLE).astype(int)
        
        # Create aggregate indicators
        df['has_source_object'] = (df[['has_source_database', 'has_source_table', 
                                       'has_source_column']].sum(axis=1) > 0).astype(int)
        df['has_target_object'] = (df[['has_target_database', 'has_target_table', 
                                       'has_target_column']].sum(axis=1) > 0).astype(int)
        
        return df
    
    def create_test_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on test type and field requirements"""
        
        # Schema tests don't need source/target columns
        df['is_schema_test'] = (df['test_family'] == 'schema').astype(int)
        
        # Allocation tests need both source and target
        df['is_allocation_test'] = (df['test_family'] == 'allocation').astype(int)
        
        # Uniqueness tests need source table/column
        df['is_uniqueness_test'] = (df['test_family'] == 'uniqueness').astype(int)
        
        # Tests requiring source-target comparison
        df['requires_source_target'] = (
            df['test_family'].isin(['allocation', 'completeness', 'referential_integrity'])
        ).astype(int)
        
        # Tests requiring only source
        df['requires_source_only'] = (
            df['test_family'].isin(['uniqueness', 'schema_drift'])
        ).astype(int)
        
        return df
    
    def create_null_result_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for clean test result handling"""
        
        # Indicator for clean test results (no violations found)
        df['has_no_violations'] = df['violation_severity'].eq(self.NO_VIOLATIONS).astype(int)
        
        # Test-type specific clean result patterns
        df['allocation_passed_clean'] = (
            (df['test_family'] == 'allocation') & 
            (df['has_no_violations'] == 1)
        ).astype(int)
        
        df['uniqueness_passed_clean'] = (
            (df['test_family'] == 'uniqueness') & 
            (df['has_no_violations'] == 1)
        ).astype(int)
        
        return df
    
    def validate_status_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate consistency between status and violation patterns"""
        
        # Check if status aligns with violations
        df['status_consistent'] = (
            ((df['status'] == 'passed') & (df['has_violations'] == 0)) |
            ((df['status'] == 'failed') & (df['has_violations'] == 1))
        ).astype(int)
        
        # Check for clean results with pass status
        df['clean_result_consistent'] = (
            (df['status'] == 'passed') & (df['has_no_violations'] == 1)
        ).astype(int)
        
        # Flag inconsistent records
        df['status_inconsistent'] = (1 - df['status_consistent']).astype(int)
        
        return df
    
    def calculate_dq_score_from_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DQ_SCORE based on status column (passes/total_tests by date)"""
        
        # Group by date and count status values
        daily_status = df.groupby('date')['status'].value_counts().unstack(fill_value=0)
        
        # Ensure we have all possible status columns
        expected_statuses = ['pass', 'fail', 'error', 'skipped']
        for status in expected_statuses:
            if status not in daily_status.columns:
                daily_status[status] = 0
        
        # Calculate DQ_SCORE (passed / total_tests)
        daily_status['total_tests'] = daily_status.sum(axis=1)
        daily_status['passed_tests'] = daily_status['pass']
        daily_status['failed_tests'] = daily_status['fail']
        daily_status['error_tests'] = daily_status['error']
        daily_status['skipped_tests'] = daily_status['skipped']
        
        # Calculate DQ_SCORE
        daily_status['DQ_SCORE'] = (
            daily_status['passed_tests'] / daily_status['total_tests']
        ).round(4)
        
        # Additional metrics
        daily_status['pass_rate'] = daily_status['DQ_SCORE']
        daily_status['fail_rate'] = (daily_status['failed_tests'] / daily_status['total_tests']).round(4)
        daily_status['error_rate'] = (daily_status['error_tests'] / daily_status['total_tests']).round(4)
        
        return daily_status[['DQ_SCORE', 'passed_tests', 'failed_tests', 'error_tests', 
                           'skipped_tests', 'total_tests', 'pass_rate', 'fail_rate', 'error_rate']]
    
    def calculate_trust_score_inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate severity-weighted trust score (lower is better)"""
        
        # Add calendar date column for proper daily grouping
        df['calendar_date'] = df.index.date
        
        # Calculate weighted failed points by calendar date
        daily_failures = df[df['status'] == 'fail'].copy()
        
        if len(daily_failures) == 0:
            # No failures, perfect trust score
            daily_dates = df['calendar_date'].unique()
            trust_scores = pd.DataFrame(index=daily_dates)
            trust_scores['TRUST_SCORE'] = 1.0  # Perfect trust when no failures
            return trust_scores
        
        # Assign severity weights
        daily_failures['severity_weight'] = daily_failures['violation_severity'].map(self.severity_weights).fillna(1)
        
        # Calculate weighted failure points by calendar date
        weighted_failures = daily_failures.groupby('calendar_date').agg({
            'severity_weight': 'sum',
            'status': 'count'  # Total failed tests
        })
        
        # Calculate actual maximum possible weighted points based on real severity distribution
        daily_severity_totals = df.groupby('calendar_date').apply(
            lambda x: (x['violation_severity'].map(self.severity_weights).fillna(0)).sum()
        ).rename('max_possible_weighted_points')
        
        # Calculate TRUST_SCORE as 1 - (weighted failure rate)
        weighted_failures = weighted_failures.join(daily_severity_totals, how='outer').fillna(0)
        weighted_failures['TRUST_SCORE'] = (
            1.0 - (weighted_failures['severity_weight'] / weighted_failures['max_possible_weighted_points'])
        ).round(4)
        
        # Handle cases with no failures
        weighted_failures['TRUST_SCORE'] = weighted_failures['TRUST_SCORE'].fillna(1.0)  # Perfect trust when no failures
        
        # Convert index back to datetime for proper joining
        weighted_failures.index = pd.to_datetime(weighted_failures.index)
        
        return weighted_failures[['TRUST_SCORE']]
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create datetime-based features from the date index"""
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Basic time features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek  # Monday=0, Sunday=6
        df['dayofyear'] = df.index.dayofyear
        df['week'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        
        # Cyclical features for better seasonality capture
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Weekend indicator
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Month start/end indicators
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features by different time periods"""
        
        # Daily aggregations (already handled by index)
        # Weekly aggregations
        weekly_stats = df.resample('W').agg({
            'test_passed': 'sum',
            'test_failed': 'sum',
            'test_error': 'sum',
            'has_violations': 'sum',
            'violation_count': 'sum',
            'execution_time': ['mean', 'sum'],
            'records_processed': 'sum'
        }).round(4)
        
        # Flatten column names
        weekly_stats.columns = ['_'.join(col).strip() for col in weekly_stats.columns.values]
        
        # Monthly aggregations
        monthly_stats = df.resample('M').agg({
            'test_passed': 'sum',
            'test_failed': 'sum',
            'test_error': 'sum',
            'has_violations': 'sum',
            'violation_count': 'sum',
            'execution_time': ['mean', 'sum'],
            'records_processed': 'sum'
        }).round(4)
        
        # Flatten column names
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        
        # Join aggregated features back to original dataframe with suffixes to avoid overlap
        df = df.join(weekly_stats, how='left', rsuffix='_weekly')
        df = df.join(monthly_stats, how='left', rsuffix='_monthly')
        
        # Fill NaN values from aggregations
        df = df.ffill().fillna(0)
        
        return df
    
    def create_severity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create severity-based features"""
        
        # Severity indicators
        df['is_high_severity'] = (df['violation_severity'] == 'HIGH SEVERITY').astype(int)
        df['is_medium_severity'] = (df['violation_severity'] == 'MEDIUM SEVERITY').astype(int)
        df['is_low_severity'] = (df['violation_severity'] == 'LOW SEVERITY').astype(int)
        df['is_no_violations'] = (df['violation_severity'] == self.NO_VIOLATIONS).astype(int)
        
        # Severity weight mapping
        df['severity_weight'] = df['violation_severity'].map(self.severity_weights).fillna(0)
        
        # Test family by severity interactions
        for test_family in df['test_family'].unique():
            if test_family != self.NOT_APPLICABLE:
                family_mask = df['test_family'] == test_family
                df[f'{test_family}_high_severity'] = (family_mask & df['is_high_severity']).astype(int)
                df[f'{test_family}_medium_severity'] = (family_mask & df['is_medium_severity']).astype(int)
                df[f'{test_family}_low_severity'] = (family_mask & df['is_low_severity']).astype(int)
        
        return df
    
    def create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance-related features"""
        
        # Execution performance metrics
        df['execution_time_per_record'] = np.where(
            df['records_processed'] > 0,
            df['execution_time'] / df['records_processed'],
            0
        )
        
        # Failure rate metrics
        df['failure_rate'] = np.where(
            df['records_processed'] > 0,
            df['records_failed'] / df['records_processed'],
            0
        )
        
        # Test complexity indicators
        df['has_source_and_target'] = (
            (df['has_source_object'] == 1) & (df['has_target_object'] == 1)
        ).astype(int)
        
        df['complexity_score'] = (
            df['has_source_and_target'] * 2 +
            df['has_source_object'] * 1 +
            df['has_target_object'] * 1 +
            df['has_violations'] * 1
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        
        # Status × test family interactions
        for test_family in df['test_family'].unique():
            if test_family != self.NOT_APPLICABLE:
                df[f'{test_family}_passed_rate'] = (
                    df.groupby('date')[f'{test_family}_passed'].transform('mean')
                )
                df[f'{test_family}_failed_rate'] = (
                    df.groupby('date')[f'{test_family}_failed'].transform('mean')
                )
        
        # Violation × object interactions
        df['violations_with_source'] = (
            df['has_violations'] * df['has_source_object']
        )
        df['violations_with_target'] = (
            df['has_violations'] * df['has_target_object']
        )
        df['violations_with_both'] = (
            df['has_violations'] * df['has_source_and_target']
        )
        
        # Performance × severity interactions
        df['high_severity_performance_impact'] = (
            df['is_high_severity'] * df['execution_time']
        )
        df['violation_count_performance_impact'] = (
            df['violation_count'] * df['execution_time_per_record']
        )
        
        return df
    
    def create_temporal_lag_features(self, df: pd.DataFrame, lag_days: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create temporal lag features for DQ_SCORE and TRUST_SCORE.
        
        Args:
            df: DataFrame with DQ_SCORE and TRUST_SCORE columns
            lag_days: List of lag periods to create
            
        Returns:
            DataFrame with lag features added
        """
        logger.info(f"Creating temporal lag features for {lag_days} days...")
        
        # Ensure dataframe is sorted by date
        df = df.sort_index()
        
        # Create lag features for DQ_SCORE
        for lag in lag_days:
            df[f'DQ_SCORE_lag_{lag}d'] = df['DQ_SCORE'].shift(lag)
            df[f'DQ_SCORE_change_{lag}d'] = df['DQ_SCORE'] - df[f'DQ_SCORE_lag_{lag}d']
            df[f'DQ_SCORE_pct_change_{lag}d'] = df['DQ_SCORE'].pct_change(lag)
        
        # Create lag features for TRUST_SCORE
        for lag in lag_days:
            df[f'TRUST_SCORE_lag_{lag}d'] = df['TRUST_SCORE'].shift(lag)
            df[f'TRUST_SCORE_change_{lag}d'] = df['TRUST_SCORE'] - df[f'TRUST_SCORE_lag_{lag}d']
            df[f'TRUST_SCORE_pct_change_{lag}d'] = df['TRUST_SCORE'].pct_change(lag)
        
        # Create rolling window features
        rolling_windows = [7, 14, 30]
        
        # Rolling statistics for DQ_SCORE
        for window in rolling_windows:
            df[f'DQ_SCORE_rolling_mean_{window}d'] = df['DQ_SCORE'].rolling(window=window).mean()
            df[f'DQ_SCORE_rolling_std_{window}d'] = df['DQ_SCORE'].rolling(window=window).std()
            df[f'DQ_SCORE_rolling_min_{window}d'] = df['DQ_SCORE'].rolling(window=window).min()
            df[f'DQ_SCORE_rolling_max_{window}d'] = df['DQ_SCORE'].rolling(window=window).max()
            df[f'DQ_SCORE_rolling_trend_{window}d'] = (
                df['DQ_SCORE'] - df[f'DQ_SCORE_rolling_mean_{window}d']
            )
        
        # Rolling statistics for TRUST_SCORE
        for window in rolling_windows:
            df[f'TRUST_SCORE_rolling_mean_{window}d'] = df['TRUST_SCORE'].rolling(window=window).mean()
            df[f'TRUST_SCORE_rolling_std_{window}d'] = df['TRUST_SCORE'].rolling(window=window).std()
            df[f'TRUST_SCORE_rolling_min_{window}d'] = df['TRUST_SCORE'].rolling(window=window).min()
            df[f'TRUST_SCORE_rolling_max_{window}d'] = df['TRUST_SCORE'].rolling(window=window).max()
            df[f'TRUST_SCORE_rolling_trend_{window}d'] = (
                df['TRUST_SCORE'] - df[f'TRUST_SCORE_rolling_mean_{window}d']
            )
        
        # Create momentum features
        df['DQ_SCORE_momentum_7d'] = df['DQ_SCORE'].pct_change(7)
        df['DQ_SCORE_momentum_14d'] = df['DQ_SCORE'].pct_change(14)
        df['TRUST_SCORE_momentum_7d'] = df['TRUST_SCORE'].pct_change(7)
        df['TRUST_SCORE_momentum_14d'] = df['TRUST_SCORE'].pct_change(14)
        
        # Create volatility features
        df['DQ_SCORE_volatility_7d'] = df['DQ_SCORE'].rolling(7).std()
        df['DQ_SCORE_volatility_14d'] = df['DQ_SCORE'].rolling(14).std()
        df['TRUST_SCORE_volatility_7d'] = df['TRUST_SCORE'].rolling(7).std()
        df['TRUST_SCORE_volatility_14d'] = df['TRUST_SCORE'].rolling(14).std()
        
        # Fill NaN values created by lag operations
        df = df.bfill().fillna(0)
        
        logger.info(f"Created {len([col for col in df.columns if 'lag_' in col or 'rolling_' in col or 'momentum' in col or 'volatility' in col])} temporal features")
        
        return df
    
    def create_dq_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create healthcare-specific DQ_SCORE categories"""
        
        # DQ_SCORE Healthcare Categories
        conditions = [
            df['DQ_SCORE'] >= 0.98,
            df['DQ_SCORE'] >= 0.92,
            df['DQ_SCORE'] >= 0.80,
            df['DQ_SCORE'] >= 0.60,
            df['DQ_SCORE'] < 0.60
        ]
        choices = [
            'CLINICAL_EXCELLENCE',
            'HIGH_QUALITY', 
            'ACCEPTABLE',
            'NEEDS_ATTENTION',
            'UNACCEPTABLE'
        ]
        df['DQ_SCORE_CATEGORY'] = np.select(conditions, choices, default='UNKNOWN')
        
        # TRUST_SCORE Risk Categories (FIXED - higher is better)
        trust_conditions = [
            df['TRUST_SCORE'] >= 0.90,
            df['TRUST_SCORE'] >= 0.75,
            df['TRUST_SCORE'] >= 0.50,
            df['TRUST_SCORE'] >= 0.25,
            df['TRUST_SCORE'] < 0.25
        ]
        trust_choices = [
            'EXCELLENT_TRUST',
            'GOOD_TRUST',
            'MODERATE_TRUST', 
            'POOR_TRUST',
            'CRITICAL_RISK'
        ]
        df['TRUST_SCORE_CATEGORY'] = np.select(trust_conditions, trust_choices, default='UNKNOWN')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info("Starting comprehensive feature engineering pipeline...")
        
        # Extract JSON features with status from test_result
        logger.info("Extracting JSON features...")
        json_features = self.extract_json_features_with_status(df)
        
        # Add datetime features
        logger.info("Creating datetime features...")
        json_features = self.create_datetime_features(json_features)
        
        # Add status-based features
        logger.info("Creating status features...")
        json_features = self.create_status_features(json_features)
        
        # Add null result specific features
        logger.info("Creating null result features...")
        json_features = self.create_null_result_features(json_features)
        
        # Validate status consistency (moved after null result features are created)
        logger.info("Validating status consistency...")
        json_features = self.validate_status_consistency(json_features)
        
        # Add missing indicators for optional fields
        logger.info("Creating missing indicators...")
        json_features = self.create_missing_indicators(json_features)
        
        # Add test type features
        logger.info("Creating test type features...")
        json_features = self.create_test_type_features(json_features)
        
        # Add severity-based features
        logger.info("Creating severity features...")
        json_features = self.create_severity_features(json_features)
        
        # Add performance features
        logger.info("Creating performance features...")
        json_features = self.create_performance_features(json_features)
        
        # Add interaction features
        logger.info("Creating interaction features...")
        json_features = self.create_interaction_features(json_features)
        
        # Add aggregation features
        logger.info("Creating aggregation features...")
        json_features = self.create_aggregation_features(json_features)
        
        logger.info(f"Feature engineering completed. Generated {len(json_features.columns)} features.")
        return json_features
    
    def create_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create final dataset with DQ_SCORE, TRUST_SCORE, and temporal lag features at daily level"""
        logger.info("Creating final dataset with scores and temporal features...")
        
        # Create daily-level dataset by aggregating features
        daily_features = self.create_daily_aggregated_features(df)
        
        # Calculate DQ_SCORE and TRUST_SCORE using aggregated daily test counts
        logger.info("Calculating DQ_SCORE and TRUST_SCORE from aggregated daily counts...")
        
        # Use the aggregated test counts to calculate DQ_SCORE
        passed_tests = daily_features['daily_test_passed_sum']
        failed_tests = daily_features['daily_test_failed_sum']
        total_tests = daily_features['daily_total_tests']
        
        # Calculate DQ_SCORE as passed_tests / total_tests
        dq_scores = passed_tests / total_tests.replace(0, np.nan)  # Avoid division by zero
        dq_scores = dq_scores.fillna(0.0)  # Set to 0 where no tests
        
        # Calculate TRUST_SCORE using the original severity-weighted method
        trust_scores = self.calculate_trust_score_inverse(df)
        
        # Add scores to daily features
        daily_features['DQ_SCORE'] = dq_scores
        daily_features['TRUST_SCORE'] = trust_scores['TRUST_SCORE']
        
        # Add test count columns for reference
        daily_features['passed_tests'] = passed_tests
        daily_features['failed_tests'] = failed_tests
        daily_features['total_tests'] = total_tests
        daily_features['pass_rate'] = dq_scores
        daily_features['fail_rate'] = failed_tests / total_tests.replace(0, np.nan).fillna(0.0)
        daily_features['error_rate'] = daily_features.get('daily_test_error_sum', 0) / total_tests.replace(0, np.nan).fillna(0.0)
        daily_features['error_tests'] = daily_features.get('daily_test_error_sum', 0)
        daily_features['skipped_tests'] = daily_features.get('daily_test_skipped_sum', 0)
        
        final_df = daily_features
        
        # Create categories
        final_df = self.create_dq_categories(final_df)
        
        # Fill any missing scores
        final_df['DQ_SCORE'] = final_df['DQ_SCORE'].fillna(0.0)
        final_df['TRUST_SCORE'] = final_df['TRUST_SCORE'].fillna(0.0)
        
        # Add temporal lag features
        logger.info("Adding temporal lag features...")
        final_df = self.create_temporal_lag_features(final_df)
        
        logger.info("Final dataset creation completed.")
        return final_df
    
    def create_daily_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create daily-level aggregated features from individual test records"""
        logger.info("Creating daily aggregated features...")
        
        # Add calendar date column for proper daily grouping
        df['calendar_date'] = df.index.date
        
        # Group by calendar date and aggregate numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove individual test-level indicators that don't make sense to aggregate,
        # but KEEP test result columns for DQ_SCORE calculation
        exclude_cols = []  # Include all columns for proper aggregation
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Aggregate numeric features by calendar date (sum, mean, max, min)
        daily_features = pd.DataFrame()
        
        for col in numeric_cols:
            if col in df.columns:
                # Use different aggregation based on feature type
                if col in ['test_passed', 'test_failed', 'test_error', 'test_skipped'] or 'count' in col or 'has_' in col or col.endswith('_passed') or col.endswith('_failed'):
                    # Sum counts and indicators
                    daily_features[f'daily_{col}_sum'] = df.groupby('calendar_date')[col].sum()
                elif 'rate' in col or 'ratio' in col or 'score' in col:
                    # Average rates and scores
                    daily_features[f'daily_{col}_mean'] = df.groupby('calendar_date')[col].mean()
                    daily_features[f'daily_{col}_max'] = df.groupby('calendar_date')[col].max()
                    daily_features[f'daily_{col}_min'] = df.groupby('calendar_date')[col].min()
                else:
                    # Default to mean for other numeric features
                    daily_features[f'daily_{col}_mean'] = df.groupby('calendar_date')[col].mean()
        
        # Add test family counts
        test_family_counts = df.groupby('calendar_date')['test_family'].value_counts().unstack(fill_value=0)
        for family in test_family_counts.columns:
            daily_features[f'daily_{family}_count'] = test_family_counts[family]
        
        # Add basic daily statistics
        daily_stats = df.groupby('calendar_date').agg({
            'status': 'count',  # Total tests per day
            'execution_time': ['mean', 'sum'],  # Execution time stats
            'records_processed': 'sum'  # Total records processed
        }).round(2)
        
        # Flatten column names
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
        daily_features = daily_features.join(daily_stats, how='left')
        
        # Rename total tests column
        if 'status_count' in daily_features.columns:
            daily_features = daily_features.rename(columns={'status_count': 'daily_total_tests'})
        
        # Convert calendar_date back to datetime index
        daily_features.index = pd.to_datetime(daily_features.index)
        
        logger.info(f"Created daily aggregated features: {len(daily_features.columns)} columns")
        return daily_features
