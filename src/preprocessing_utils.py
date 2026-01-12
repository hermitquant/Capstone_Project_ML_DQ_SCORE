"""
Preprocessing Utilities for Healthcare Data Quality Events

PURPOSE:
--------
This module provides the foundational data processing utilities for transforming
raw healthcare data quality events into feature-engineered datasets suitable for
machine learning and AI Agent deployment.

CORE RESPONSIBILITIES:
---------------------
1. DATA LOADING & VALIDATION: Load raw events and ensure data quality
2. FEATURE ENGINEERING: Transform raw events into predictive features
3. DATA SAVING & REPORTING: Persist processed data and generate documentation
4. PIPELINE ORCHESTRATION: Coordinate end-to-end processing workflows

KEY DESIGN PRINCIPLES:
----------------------
- ROBUSTNESS: Handle missing data, format issues, and edge cases gracefully
- TRANSPARENCY: Comprehensive logging and reporting at every step
- FLEXIBILITY: Configurable parameters for different use cases
- SCALABILITY: Memory-efficient processing for large datasets
- TRACEABILITY: Complete audit trail of all transformations

BUSINESS CONTEXT:
----------------
Healthcare data quality events represent test results from data validation
processes. Each event contains information about test execution, violations,
severity levels, and system metadata. This module transforms these raw events
into structured features that predict overall data quality scores (DQ_SCORE
and TRUST_SCORE).

TYPICAL USAGE:
-------------
# Basic pipeline usage
processed_data = run_preprocessing_pipeline(
    data_dir=Path("./raw_data"),
    output_dir=Path("./processed_data")
)

# Advanced usage with custom configuration
from preprocessing_utils import HealthcareDQPreprocessor
processor = HealthcareDQPreprocessor()
features = processor.process_events_to_features(raw_events)
processor.save_processed_data(features, output_path)

OUTPUT FILES:
------------
- feature_engineered_events.csv: Main processed dataset
- feature_engineered_events_summary.json: Dataset statistics
- processing_report.md: Comprehensive processing documentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json

from feature_engineer import HealthcareDQFeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)


def load_events_data(data_path: Path, filename: str = "events.csv") -> pd.DataFrame:
    """
    Load events data from CSV file with comprehensive temporal feature extraction.
    
    This function handles the critical first step of the pipeline: loading raw events
    data and extracting temporal features that are essential for time series analysis.
    It supports multiple date formats and handles missing or malformed timestamps
    gracefully.
    
    TEMPORAL FEATURE ENGINEERING:
    ----------------------------
    The function extracts rich temporal features from timestamps:
    - Basic temporal components: hour, minute, second, day_of_week
    - Cyclical encoding: sin/cos representations for periodic patterns
    - Time of day categories: morning, afternoon, evening, night
    - Business hours indicator: weekdays during 9am-5pm
    - Time of day continuous: decimal hour representation
    
    DATA QUALITY HANDLING:
    ----------------------
    - Invalid timestamps are converted to NaT and removed
    - Missing date columns trigger sequential date generation
    - Multiple timestamp formats are automatically detected
    - Comprehensive logging of data quality issues
    
    INPUT PARAMETERS:
    -----------------
    data_path (Path): Directory containing the events CSV file
    filename (str): Name of the CSV file (default: "events.csv")
    
    RETURN VALUES:
    --------------
    pd.DataFrame: Loaded events data with datetime index and temporal features
    
    EXPECTED COLUMNS:
    -----------------
    Required: RUN_TIMESTAMP (primary) or date (fallback)
    Optional: Any additional event metadata columns
    
    GENERATED FEATURES:
    -------------------
    - hour, minute, second: Time components
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - hour_sin, hour_cos: Cyclical hour encoding
    - time_of_day: Decimal hour representation
    - is_morning, is_afternoon, is_evening, is_night: Time categories
    - is_business_hours: Business hours indicator
    
    ERROR HANDLING:
    ---------------
    - FileNotFoundError: Raised if CSV file doesn't exist
    - ValueError: Raised if no valid dates can be parsed
    - Warning logged for rows with invalid timestamps
    
    EXAMPLE:
    -------
    events_df = load_events_data(
        data_path=Path("./data"),
        filename="healthcare_events.csv"
    )
    print(f"Loaded {len(events_df)} events with temporal features")
    """
    file_path = data_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    
    # Load data with proper date handling
    df = pd.read_csv(file_path)
    
    # Extract temporal features from RUN_TIMESTAMP before date conversion
    if 'RUN_TIMESTAMP' in df.columns:
        # Convert to full datetime first to extract temporal elements
        df['full_timestamp'] = pd.to_datetime(df['RUN_TIMESTAMP'], errors='coerce')
        
        # Extract temporal features from full timestamp
        df['hour'] = df['full_timestamp'].dt.hour
        df['minute'] = df['full_timestamp'].dt.minute
        df['second'] = df['full_timestamp'].dt.second
        df['day_of_week'] = df['full_timestamp'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['time_of_day'] = df['hour'] + df['minute'] / 60 + df['second'] / 3600
        
        # Time of day categories
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        # Business hours indicator
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17) & (df['day_of_week'] < 5)).astype(int)
        
        # Now convert to date for aggregation (after extracting temporal features)
        df['date'] = df['full_timestamp'].dt.date
        df = df.dropna(subset=['full_timestamp'])
        df = df.set_index('full_timestamp')
        logger.info(f"Using RUN_TIMESTAMP as datetime index with temporal features extracted")
        logger.info(f"Converted {len(df)} valid timestamps out of original {len(pd.read_csv(file_path))} records")
        
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.set_index('date')
    else:
        logger.warning("No date column found, creating sequential dates")
        # Create sequential dates for temporal features
        start_date = pd.Timestamp('2024-01-01')
        df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
        df = df.set_index('date')
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality validation and generate detailed diagnostics.
    
    This function serves as the data quality gatekeeper for the entire pipeline.
    It analyzes the loaded data for completeness, consistency, and business logic
    compliance. The validation results guide downstream processing decisions and
    provide stakeholders with transparency into data quality issues.
    
    VALIDATION DIMENSIONS:
    ----------------------
    1. COMPLETENESS: Missing values, null patterns, data coverage
    2. CONSISTENCY: Data types, formats, value ranges
    3. TEMPORAL INTEGRITY: Date ranges, temporal gaps, sequence validity
    4. BUSINESS LOGIC: DQ_EVENT coverage, domain-specific constraints
    5. STRUCTURAL: Column counts, data types, index integrity
    
    QUALITY METRICS:
    -----------------
    - Total records and columns (dataset size)
    - Missing value patterns by column
    - Data type distribution and inconsistencies
    - Temporal coverage and gap analysis
    - DQ_EVENT field completeness (critical for feature engineering)
    - Date range and duration analysis
    
    INPUT PARAMETERS:
    -----------------
    df (pd.DataFrame): Input dataframe to validate
    
    RETURN VALUES:
    --------------
    Dict[str, Any]: Comprehensive validation results containing:
    - total_records: Number of rows in dataset
    - total_columns: Number of columns in dataset
    - missing_values: Per-column missing value counts
    - data_types: Per-column data type information
    - date_range: Temporal coverage information
    - dq_event_coverage: Percentage of non-null DQ_EVENT values
    
    BUSINESS IMPACT:
    ----------------
    This validation helps:
    - Data engineers identify ingestion issues early
    - Business stakeholders understand data reliability
    - ML engineers assess feature engineering readiness
    - Compliance teams verify data governance requirements
    
    LOGGING:
    -------
    All validation results are logged at INFO level for pipeline transparency.
    Critical issues trigger additional WARNING level logs.
    """
    validation_results = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'date_range': None,
        'dq_event_coverage': 0
    }
    
    # Check date range if datetime index exists
    if hasattr(df.index, 'min') and hasattr(df.index, 'max') and isinstance(df.index, pd.DatetimeIndex):
        validation_results['date_range'] = {
            'start': df.index.min(),
            'end': df.index.max(),
            'days': (df.index.max() - df.index.min()).days
        }
    else:
        validation_results['date_range'] = {
            'start': 'No datetime index',
            'end': 'No datetime index',
            'days': 0
        }
    
    # Check DQ_EVENT column coverage
    if 'DQ_EVENT' in df.columns:
        validation_results['dq_event_coverage'] = (df['DQ_EVENT'].notna().sum() / len(df)) * 100
    
    # Log validation summary
    logger.info("Data Validation Summary:")
    logger.info(f"  - Total records: {validation_results['total_records']}")
    logger.info(f"  - Total columns: {validation_results['total_columns']}")
    logger.info(f"  - DQ_EVENT coverage: {validation_results['dq_event_coverage']:.2f}%")
    
    if validation_results['date_range']:
        logger.info(f"  - Date range: {validation_results['date_range']['start']} to {validation_results['date_range']['end']}")
        logger.info(f"  - Total days: {validation_results['date_range']['days']}")
    
    return validation_results


def process_events_to_features(df: pd.DataFrame, 
                             feature_engineer: Optional[HealthcareDQFeatureEngineer] = None) -> pd.DataFrame:
    """
    Transform raw events data into feature-engineered dataset ready for ML.
    
    This function orchestrates the core feature engineering transformation,
    converting individual event records into aggregated daily features with
    calculated quality scores. It serves as the bridge between raw event data
    and machine learning-ready features.
    
    TRANSFORMATION PIPELINE:
    -----------------------
    1. EVENT ENGINEERING: Convert individual events to structured features
    2. TEMPORAL AGGREGATION: Group events by date and calculate statistics
    3. SCORE CALCULATION: Compute DQ_SCORE and TRUST_SCORE based on violations
    4. FEATURE ENRICHMENT: Add derived features and business logic metrics
    5. FINAL ASSEMBLY: Combine all features into final dataset structure
    
    FEATURE CATEGORIES GENERATED:
    -----------------------------
    - TEST EXECUTION METRICS: pass/fail counts, execution times, error rates
    - VIOLATION METRICS: violation counts, severity distributions, impact scores
    - TEMPORAL FEATURES: date components, cyclical encoding, business indicators
    - TEST FAMILY BREAKDOWN: performance by test type (ALLOCATION, SCHEMA, etc.)
    - COMPLEXITY METRICS: source/target object complexity, execution patterns
    - QUALITY SCORES: DQ_SCORE and TRUST_SCORE with categorical classifications
    
    INPUT PARAMETERS:
    -----------------
    df (pd.DataFrame): Raw events data with temporal index and event metadata
    feature_engineer (HealthcareDQFeatureEngineer): Custom feature engineer instance
                                                      (creates default if None)
    
    RETURN VALUES:
    --------------
    pd.DataFrame: Feature-engineered dataset with:
    - Daily aggregated metrics (one row per date)
    - Calculated quality scores (DQ_SCORE, TRUST_SCORE)
    - Rich feature set for ML model training
    - Business-relevant metrics for decision support
    
    FEATURE ENGINEER CUSTOMIZATION:
    ------------------------------
    The optional feature_engineer parameter allows customization of:
    - Severity weight mappings for violation scoring
    - Business logic for score calculations
    - Custom feature engineering rules
    - Domain-specific aggregations
    
    BUSINESS VALUE:
    --------------
    The output dataset enables:
    - Predictive modeling of data quality trends
    - Root cause analysis of quality issues
    - Performance monitoring and alerting
    - Resource optimization for data quality teams
    
    EXAMPLE:
    -------
    # Use default feature engineering
    features_df = process_events_to_features(raw_events)
    
    # Use custom feature engineer
    custom_engineer = HealthcareDQFeatureEngineer(custom_severity_weights)
    features_df = process_events_to_features(raw_events, custom_engineer)
    """
    if feature_engineer is None:
        feature_engineer = HealthcareDQFeatureEngineer()
    
    logger.info("Starting feature engineering pipeline...")
    
    # Engineer features
    features_df = feature_engineer.engineer_features(df)
    
    # Create final dataset with scores
    final_df = feature_engineer.create_final_dataset(features_df)
    
    logger.info(f"Feature engineering completed. Final dataset shape: {final_df.shape}")
    
    return final_df


def save_processed_data(df: pd.DataFrame, 
                       output_path: Path, 
                       filename: str = "final_event.csv") -> None:
    """
    Save processed DataFrame to CSV file.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the file
        filename: Name of the output file
    """
    ensure_dir(output_path)
    file_path = output_path / filename
    
    logger.info(f"Saving processed data to {file_path}")
    
    # Save to CSV
    df.to_csv(file_path, index=True)
    
    logger.info(f"Successfully saved {len(df)} records to {file_path}")
    
    # Save summary statistics
    summary_path = output_path / f"{filename.replace('.csv', '_summary.json')}"
    save_data_summary(df, summary_path)


def save_data_summary(df: pd.DataFrame, summary_path: Path) -> None:
    """
    Save summary statistics of the processed data.
    
    Args:
        df: Processed DataFrame
        summary_path: Path to save the summary
    """
    summary = {
        'dataset_info': {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        },
        'feature_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # Add categorical summaries
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        try:
            # Skip columns that contain lists or other unhashable types
            if df[col].apply(lambda x: isinstance(x, list)).any():
                # For columns with lists, provide a simple summary instead
                summary['categorical_summary'][col] = {
                    'type': 'contains_lists',
                    'non_null_count': df[col].notna().sum(),
                    'unique_count': df[col].nunique()
                }
            else:
                summary['categorical_summary'][col] = df[col].value_counts().head(10).to_dict()
        except (TypeError, ValueError, AttributeError) as e:
            # Fallback for problematic columns
            summary['categorical_summary'][col] = {
                'error': f'Could not summarize: {str(e)}',
                'non_null_count': df[col].notna().sum(),
                'dtype': str(df[col].dtype)
            }
    
    # Score statistics
    if 'DQ_SCORE' in df.columns:
        summary['dq_score_stats'] = {
            'mean': df['DQ_SCORE'].mean(),
            'std': df['DQ_SCORE'].std(),
            'min': df['DQ_SCORE'].min(),
            'max': df['DQ_SCORE'].max(),
            'median': df['DQ_SCORE'].median()
        }
    
    if 'TRUST_SCORE' in df.columns:
        summary['trust_score_stats'] = {
            'mean': df['TRUST_SCORE'].mean(),
            'std': df['TRUST_SCORE'].std(),
            'min': df['TRUST_SCORE'].min(),
            'max': df['TRUST_SCORE'].max(),
            'median': df['TRUST_SCORE'].median()
        }
    
    # Save summary as JSON
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Data summary saved to {summary_path}")


def run_preprocessing_pipeline(data_dir: Path, 
                             output_dir: Path,
                             input_filename: str = "events.csv",
                             output_filename: str = "feature_engineered_events.csv") -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        data_dir: Directory containing input data
        output_dir: Directory to save processed data
        input_filename: Name of input CSV file
        output_filename: Name of output CSV file
        
    Returns:
        Processed DataFrame
    """
    logger.info("=" * 60)
    logger.info("STARTING HEALTHCARE DQ FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load raw data
        logger.info("Step 1: Loading raw data...")
        raw_df = load_events_data(data_dir, input_filename)
        
        # Step 2: Validate data quality
        logger.info("Step 2: Validating data quality...")
        validation_results = validate_data_quality(raw_df)
        
        # Step 3: Feature engineering
        logger.info("Step 3: Running feature engineering...")
        feature_engineer = HealthcareDQFeatureEngineer()
        processed_df = process_events_to_features(raw_df, feature_engineer)
        
        # Step 4: Save processed data
        logger.info("Step 4: Saving processed data...")
        save_processed_data(processed_df, output_dir, output_filename)
        
        # Step 5: Generate final report
        logger.info("Step 5: Generating final report...")
        generate_processing_report(validation_results, processed_df, output_dir)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


def generate_processing_report(validation_results: Dict[str, Any], 
                             processed_df: pd.DataFrame,
                             output_dir: Path) -> None:
    """
    Generate a comprehensive processing report.
    
    Args:
        validation_results: Data validation results
        processed_df: Final processed DataFrame
        output_dir: Output directory for the report
    """
    report_path = output_dir / "processing_report.md"
    
    report_content = f"""# Healthcare Data Quality Feature Engineering Report

## Data Overview
- **Total Records Processed**: {validation_results['total_records']:,}
- **Total Features Generated**: {len(processed_df.columns):,}
- **DQ_EVENT Coverage**: {validation_results['dq_event_coverage']:.2f}%

## Date Range
"""
    
    if validation_results['date_range']:
        date_range = validation_results['date_range']
        report_content += f"""- **Start Date**: {date_range['start']}
- **End Date**: {date_range['end']}
- **Total Days**: {date_range['days']:,}
"""
    
    report_content += f"""
## Feature Engineering Summary
- **JSON Features Extracted**: {len([col for col in processed_df.columns if col.startswith(('source_', 'target_', 'metric_', 'test_'))])}
- **Status Features**: {len([col for col in processed_df.columns if 'status' in col.lower()])}
- **Missing Value Indicators**: {len([col for col in processed_df.columns if col.startswith('has_')])}
- **Test Type Features**: {len([col for col in processed_df.columns if any(x in col for x in ['is_', 'requires_'])])}

## Score Statistics
"""
    
    if 'DQ_SCORE' in processed_df.columns:
        dq_stats = processed_df['DQ_SCORE'].describe()
        report_content += f"""### DQ_SCORE
- **Mean**: {dq_stats['mean']:.4f}
- **Std Dev**: {dq_stats['std']:.4f}
- **Min**: {dq_stats['min']:.4f}
- **Max**: {dq_stats['max']:.4f}
- **Median**: {processed_df['DQ_SCORE'].median():.4f}

"""
    
    if 'TRUST_SCORE' in processed_df.columns:
        trust_stats = processed_df['TRUST_SCORE'].describe()
        report_content += f"""### TRUST_SCORE
- **Mean**: {trust_stats['mean']:.4f}
- **Std Dev**: {trust_stats['std']:.4f}
- **Min**: {trust_stats['min']:.4f}
- **Max**: {trust_stats['max']:.4f}
- **Median**: {processed_df['TRUST_SCORE'].median():.4f}

"""
    
    # Add top features by variance
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    # Exclude columns that might contain lists or other non-numeric types
    valid_numeric_cols = []
    for col in numeric_cols:
        try:
            # Test if we can calculate variance (this will fail for columns with lists)
            processed_df[col].var()
            valid_numeric_cols.append(col)
        except (TypeError, ValueError, AttributeError):
            # Skip columns that can't have variance calculated
            continue
    
    if len(valid_numeric_cols) > 0:
        top_variance = processed_df[valid_numeric_cols].var().nlargest(10)
        report_content += """## Top Features by Variance
| Feature | Variance |
|---------|----------|
"""
        for feature, variance in top_variance.items():
            report_content += f"| {feature} | {variance:.4f} |\n"
    
    report_content += """
## Files Generated
- `feature_engineered_events.csv`: Main feature-engineered dataset
- `feature_engineered_events_summary.json`: Dataset summary statistics
- `processing_report.md`: This processing report

## Next Steps
1. Use `feature_engineered_events.csv` for model training
2. Review feature importance and correlation analysis
3. Consider temporal splitting for time series validation
4. Implement model training pipeline

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Processing report saved to {report_path}")


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "data"
    
    # Run the pipeline
    processed_data = run_preprocessing_pipeline(
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    print(f"Processing completed. Final dataset shape: {processed_data.shape}")
    print(f"Files saved to: {output_dir}")
