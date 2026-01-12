"""
Updated Preprocessing Pipeline with Encoding Support
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from typing import Tuple, Dict, Any

class DatasetEncoder:
    """
    Handle encoding for the DQ dataset with minimal categorical features
    """
    
    def __init__(self):
        self.categorical_features = ['DQ_SCORE_CATEGORY', 'TRUST_SCORE_CATEGORY']
        self.encoders = {}
        self.feature_names = []
    
    def prepare_for_regression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset for regression - drop categorical target derivatives
        """
        print('=== PREPARING DATASET FOR REGRESSION ===')
        
        # Check which categorical features exist
        existing_cat_features = [col for col in self.categorical_features if col in df.columns]
        
        if existing_cat_features:
            print(f'Found categorical features: {existing_cat_features}')
            print('Dropping categorical target derivatives for regression...')
            
            df_regression = df.drop(columns=existing_cat_features)
            
            print(f'Original features: {df.shape[1]}')
            print(f'After dropping categoricals: {df_regression.shape[1]}')
            print(f'All remaining features: Numeric ✅')
            
            return df_regression
        else:
            print('No categorical features found - dataset ready for regression')
            return df.copy()
    
    def prepare_for_classification(self, df: pd.DataFrame, 
                                  target_category: str = 'DQ_SCORE_CATEGORY') -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare dataset for classification - encode categorical targets
        """
        print(f'=== PREPARING DATASET FOR CLASSIFICATION ===')
        print(f'Target category: {target_category}')
        
        if target_category not in df.columns:
            raise ValueError(f'Target category {target_category} not found in dataset')
        
        df_encoded = df.copy()
        
        # Define ordinal mappings based on business logic
        dq_mapping = {
            'UNACCEPTABLE': 0,
            'NEEDS_ATTENTION': 1, 
            'ACCEPTABLE': 2,
            'CLINICAL_EXCELLENCE': 3
        }
        
        trust_mapping = {
            'CRITICAL_RISK': 0,
            'POOR_TRUST': 1,
            'MODERATE_TRUST': 2,
            'GOOD_TRUST': 3,
            'EXCELLENT_TRUST': 4
        }
        
        # Apply encoding
        if target_category == 'DQ_SCORE_CATEGORY':
            df_encoded[f'{target_category}_encoded'] = df[target_category].map(dq_mapping)
            mapping_used = dq_mapping
        elif target_category == 'TRUST_SCORE_CATEGORY':
            df_encoded[f'{target_category}_encoded'] = df[target_category].map(trust_mapping)
            mapping_used = trust_mapping
        else:
            raise ValueError(f'Unknown target category: {target_category}')
        
        # Drop original categorical columns
        categorical_cols = [col for col in self.categorical_features if col in df_encoded.columns]
        df_encoded = df_encoded.drop(columns=categorical_cols)
        
        encoding_info = {
            'target_category': target_category,
            'mapping': mapping_used,
            'encoded_column': f'{target_category}_encoded',
            'classes': list(mapping_used.keys())
        }
        
        print(f'Encoded {target_category} with {len(mapping_used)} classes')
        print(f'Class mapping: {mapping_used}')
        print(f'Final dataset shape: {df_encoded.shape}')
        
        return df_encoded, encoding_info
    
    def one_hot_encode_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode all categorical features (if needed for some models)
        """
        print('=== ONE-HOT ENCODING CATEGORICAL FEATURES ===')
        
        # Find existing categorical features
        existing_cat_features = [col for col in self.categorical_features if col in df.columns]
        
        if not existing_cat_features:
            print('No categorical features found')
            return df.copy()
        
        # Separate numeric and categorical
        numeric_df = df.drop(columns=existing_cat_features)
        categorical_df = df[existing_cat_features]
        
        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_array = encoder.fit_transform(categorical_df)
        
        # Create feature names
        feature_names = []
        for i, col in enumerate(existing_cat_features):
            categories = encoder.categories_[i][1:]  # Drop first to avoid multicollinearity
            for cat in categories:
                feature_names.append(f'{col}_{cat}')
        
        # Create encoded DataFrame
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
        
        # Combine
        final_df = pd.concat([numeric_df, encoded_df], axis=1)
        
        print(f'One-hot encoded {len(existing_cat_features)} features')
        print(f'Created {len(feature_names)} encoded features')
        print(f'Final dataset shape: {final_df.shape}')
        
        return final_df

def create_ml_ready_dataset(file_path: str, 
                           task_type: str = 'regression',
                           target_category: str = 'DQ_SCORE_CATEGORY') -> Tuple[pd.DataFrame, Any]:
    """
    Create ML-ready dataset with appropriate encoding
    
    Args:
        file_path: Path to the irregular dataset
        task_type: 'regression' or 'classification'
        target_category: Which category to predict (for classification)
    
    Returns:
        Tuple of (prepared_dataset, encoding_info)
    """
    print('=== CREATING ML-READY DATASET ===')
    
    # Load dataset
    df = pd.read_csv(file_path, index_col='calendar_date', parse_dates=True)
    print(f'Loaded dataset: {df.shape}')
    
    # Initialize encoder
    encoder = DatasetEncoder()
    
    if task_type == 'regression':
        # For regression, drop categorical target derivatives
        df_prepared = encoder.prepare_for_regression(df)
        encoding_info = {'task_type': 'regression', 'target': 'DQ_SCORE'}
        
    elif task_type == 'classification':
        # For classification, encode categorical targets
        df_prepared, encoding_info = encoder.prepare_for_classification(df, target_category)
        
    else:
        raise ValueError(f'Unknown task type: {task_type}')
    
    # Ensure all features are numeric
    numeric_features = df_prepared.select_dtypes(include=[np.number]).columns
    if len(numeric_features) != len(df_prepared.columns):
        non_numeric = df_prepared.select_dtypes(exclude=[np.number]).columns
        print(f'Warning: Non-numeric features found: {list(non_numeric)}')
        df_prepared = df_prepared[numeric_features]
    
    print(f'✅ Final ML-ready dataset: {df_prepared.shape}')
    print(f'✅ All features are numeric')
    print(f'✅ Ready for model training')
    
    return df_prepared, encoding_info

# Example usage
if __name__ == "__main__":
    # For regression (predict DQ_SCORE)
    df_reg, info_reg = create_ml_ready_dataset(
        '../data/feature_engineered_events_irregular.csv',
        task_type='regression'
    )
    
    print('\\n' + '='*50)
    
    # For classification (predict DQ_SCORE_CATEGORY)
    df_clf, info_clf = create_ml_ready_dataset(
        '../data/feature_engineered_events_irregular.csv',
        task_type='classification',
        target_category='DQ_SCORE_CATEGORY'
    )
