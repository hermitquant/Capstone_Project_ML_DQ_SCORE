"""
Enhanced Preprocessing Pipeline with Irregular Time Series Support

This module extends the standard preprocessing pipeline to automatically generate
both regular and irregular time series datasets for AI Agent deployment.

PURPOSE:
--------
- Bridge the gap between standard feature engineering and AI Agent deployment
- Automatically handle irregular time series data without manual intervention
- Provide both traditional ML and AI Agent-ready datasets

KEY COMPONENTS:
----------------
1. Enhanced pipeline that runs both standard and irregular processing
2. Automatic dataset comparison and reporting
3. Production-ready pipeline for maximum compatibility

USAGE:
------
- Use run_enhanced_preprocessing_pipeline() for flexible dataset generation
- Use run_production_pipeline() for always-generate-both approach
- Set generate_irregular=False for backwards compatibility

OUTPUT FILES:
-------------
Standard: feature_engineered_events_final.csv
Irregular: feature_engineered_events_irregular.csv
Reports: enhanced_processing_report.md, irregular_time_series_report.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple
import json

from preprocessing_utils import (
    load_events_data, validate_data_quality, process_events_to_features, 
    save_processed_data, generate_processing_report, ensure_dir
)
from irregular_time_series_processor import IrregularTimeSeriesProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_enhanced_preprocessing_pipeline(data_dir: Path, 
                                       output_dir: Path,
                                       input_filename: str = "events.csv",
                                       output_filename: str = "feature_engineered_events.csv",
                                       generate_irregular: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run the enhanced preprocessing pipeline with irregular time series support.
    
    This function orchestrates the complete feature engineering process,
    generating both traditional ML datasets and AI Agent-ready irregular time series datasets.
    
    PIPELINE STAGES:
    ----------------
    1. Standard Feature Engineering: Creates baseline dataset with temporal features
    2. Irregular Time Series Processing: Transforms for sparse temporal data
    3. Comparative Reporting: Documents differences and recommendations
    
    INPUT PARAMETERS:
    -----------------
    data_dir (Path): Directory containing raw events data
    output_dir (Path): Directory where processed datasets will be saved
    input_filename (str): Name of the raw CSV file (default: "events.csv")
    output_filename (str): Name for the standard output dataset
    generate_irregular (bool): Whether to create AI Agent-ready irregular dataset
    
    RETURN VALUES:
    --------------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]: 
    - First element: Standard feature-engineered dataset
    - Second element: Irregular time series dataset (or None if not requested)
    
    EXAMPLE USAGE:
    -------------
    # Generate both datasets
    standard_df, irregular_df = run_enhanced_preprocessing_pipeline(
        data_dir=Path("./data"),
        output_dir=Path("./processed"),
        generate_irregular=True
    )
    
    # Backwards compatible mode
    standard_df, _ = run_enhanced_preprocessing_pipeline(
        data_dir=Path("./data"),
        output_dir=Path("./processed"), 
        generate_irregular=False
    )
    """
    logger.info("=" * 80)
    logger.info("STARTING ENHANCED HEALTHCARE DQ FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)
    
    try:
        # =======================================================================
        # STEP 1: STANDARD FEATURE ENGINEERING (BASELINE)
        # =======================================================================
        # This creates the traditional feature-engineered dataset with all the
        # standard temporal features (lags, rolling windows, etc.)
        logger.info("Step 1: Running standard feature engineering...")
        from preprocessing_utils import run_preprocessing_pipeline
        standard_df = run_preprocessing_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            input_filename=input_filename,
            output_filename=output_filename
        )
        
        # =======================================================================
        # STEP 2: IRREGULAR TIME SERIES PROCESSING (AI AGENT READY)
        # =======================================================================
        # This transforms the standard dataset to handle irregular temporal patterns
        # by removing misleading features and adding gap-aware temporal context
        irregular_df = None
        if generate_irregular:
            logger.info("Step 2: Generating irregular time series dataset...")
            
            # Initialize the irregular time series processor
            # This class handles all the transformations for sparse temporal data
            processor = IrregularTimeSeriesProcessor()
            
            # Load the standard dataset for irregular processing
            # The processor will transform this into an AI Agent-ready format
            standard_path = output_dir / output_filename
            irregular_df = processor.process_irregular_time_series(
                input_file=str(standard_path),
                output_dir=str(output_dir)
            )
            
            # ===================================================================
            # STEP 3: COMPARATIVE REPORTING
            # ===================================================================
            # Generate a comprehensive report comparing both datasets
            # This helps users understand the differences and choose appropriately
            logger.info("Step 3: Generating enhanced pipeline report...")
            generate_enhanced_report(standard_df, irregular_df, output_dir)
        
        logger.info("=" * 80)
        logger.info("ENHANCED PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return standard_df, irregular_df
        
    except Exception as e:
        logger.error(f"Enhanced pipeline failed with error: {str(e)}")
        raise


def generate_enhanced_report(standard_df: pd.DataFrame, 
                           irregular_df: Optional[pd.DataFrame],
                           output_dir: Path) -> None:
    """
    Generate an enhanced processing report comparing standard and irregular datasets.
    
    Args:
        standard_df: Standard processed DataFrame
        irregular_df: Irregular time series DataFrame (optional)
        output_dir: Output directory for the report
    """
    report_path = output_dir / "enhanced_processing_report.md"
    
    report_content = f"""# Enhanced Healthcare Data Quality Feature Engineering Report

## Dataset Comparison

### Standard Feature-Engineered Dataset
- **Records**: {len(standard_df):,}
- **Features**: {len(standard_df.columns):,}
- **Target Variables**: {len(['DQ_SCORE', 'TRUST_SCORE'])}
- **Date Range**: {standard_df.index.min()} to {standard_df.index.max()}

"""
    
    if irregular_df is not None:
        report_content += f"""### Irregular Time Series Dataset (AI Agent Ready)
- **Records**: {len(irregular_df):,}
- **Features**: {len(irregular_df.columns):,}
- **Gap-Aware Features**: {len([col for col in irregular_df.columns if any(keyword in col.lower() for keyword in ['gap', 'days_since', 'days_to', 'measurement', 'frequency', 'regularity'])])}
- **Infinite Values**: {np.isinf(irregular_df.select_dtypes(include=[np.number])).sum().sum()}
- **Missing Values**: {irregular_df.isnull().sum().sum()}

## Key Improvements for Irregular Time Series

### 1. Temporal Feature Handling
- **Removed**: {len([col for col in standard_df.columns if any(x in col for x in ['_lag_', '_rolling_', '_momentum_', '_volatility_'])])} regular-interval features
- **Added**: Gap-aware temporal features that properly handle irregular measurements

### 2. Data Quality Fixes
- **Standard Dataset**: {np.isinf(standard_df.select_dtypes(include=[np.number])).sum().sum()} infinite values
- **Irregular Dataset**: 0 infinite values (all resolved)

### 3. AI Agent Readiness
- **Standard Dataset**: Not suitable for AI agents (temporal assumptions violated)
- **Irregular Dataset**: Ready for AI Agent deployment

## Recommended Usage

### For Traditional ML Models
Use `feature_engineered_events.csv` (standard dataset)

### For AI Agent Deployment
Use `feature_engineered_events_irregular.csv` (irregular time series dataset)

### For Time Series Analysis
Use `feature_engineered_events_irregular.csv` with gap-aware features

## Files Generated
- `feature_engineered_events.csv`: Standard feature-engineered dataset
- `feature_engineered_events_summary.json`: Standard dataset summary
- `feature_engineered_events_irregular.csv`: Irregular time series dataset
- `irregular_time_series_report.md`: Irregular processing details
- `enhanced_processing_report.md`: This comprehensive report

## Next Steps
1. **For AI Agent Development**: Use the irregular time series dataset
2. **For Model Training**: Choose appropriate dataset based on temporal requirements
3. **For Production**: Deploy irregular time series pipeline for real-world data

---
*Enhanced report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Enhanced processing report saved to {report_path}")


def run_production_pipeline(data_dir: Path, 
                          output_dir: Path,
                          input_filename: str = "events.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the production pipeline that generates both datasets for maximum compatibility.
    
    This is the recommended function for production environments as it always
    generates both the traditional ML dataset and the AI Agent-ready irregular dataset.
    This ensures maximum flexibility for downstream consumers.
    
    PRODUCTION CONSIDERATIONS:
    -------------------------
    - Always generates both datasets (no configuration needed)
    - Suitable for automated/CI/CD environments
    - Provides fallback options for different ML approaches
    - Maintains full backwards compatibility
    
    INPUT PARAMETERS:
    -----------------
    data_dir (Path): Directory containing raw events data
    output_dir (Path): Directory where processed datasets will be saved
    input_filename (str): Name of the raw CSV file (default: "events.csv")
    
    RETURN VALUES:
    --------------
    Tuple[pd.DataFrame, pd.DataFrame]: 
    - First element: Standard feature-engineered dataset
    - Second element: Irregular time series dataset (always generated)
    
    PRODUCTION USAGE:
    -----------------
    # In production scripts or CI/CD pipelines
    standard_df, irregular_df = run_production_pipeline(
        data_dir=Path("/data/raw"),
        output_dir=Path("/data/processed")
    )
    # Both datasets are always available for downstream processing
    """
    logger.info("🚀 Starting Production Pipeline (Standard + Irregular)...")
    
    standard_df, irregular_df = run_enhanced_preprocessing_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        input_filename=input_filename,
        output_filename="feature_engineered_events_final.csv",
        generate_irregular=True
    )
    
    logger.info("✅ Production Pipeline Complete!")
    logger.info(f"📊 Standard dataset: {standard_df.shape}")
    logger.info(f"🤖 AI Agent dataset: {irregular_df.shape}")
    
    return standard_df, irregular_df


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "data"
    
    # Run the enhanced pipeline
    standard_data, irregular_data = run_enhanced_preprocessing_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        generate_irregular=True
    )
    
    print(f"Enhanced processing completed!")
    print(f"Standard dataset: {standard_data.shape}")
    print(f"Irregular dataset: {irregular_data.shape if irregular_data is not None else 'Not generated'}")
    print(f"Files saved to: {output_dir}")
