# Import required libraries for data manipulation, visualization, and web app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import streamlit as st
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="DQ_SCORE Forecast",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.info-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #00b4d8;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_forecast_artifact():
    """
    Load the latest forecast model artifact from the models directory.
    
    The artifact contains:
    - Trained model (if ML model won)
    - Scaler (for linear models)
    - Feature list
    - Model metadata and performance metrics
    - Selected model name (could be 'Naive(last_value)')
    
    Returns:
        Dictionary with artifact contents, or None if no artifact found
    """
    # Get project root more robustly
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        return None
    
    # Find all forecast artifacts with naming pattern: dq_score_next_forecaster_*.pkl
    artifacts = list(models_dir.glob("dq_score_next_forecaster_*.pkl"))
    if not artifacts:
        return None
    
    # Load the most recently created artifact
    latest = max(artifacts, key=lambda x: x.stat().st_mtime)
    try:
        artifact = joblib.load(latest)
        return artifact
    except Exception as e:
        return None

@st.cache_data
def load_engineered_data():
    """
    Load the engineered dataset with caching for performance.
    
    This function:
    1. Determines the project root path robustly
    2. Loads the feature_engineered_events_irregular.csv file
    3. Sets the calendar_date column as datetime index
    4. Returns sorted data for consistent time series operations
    
    The @st.cache_data decorator caches the result to avoid reloading
    the same file on every interaction, improving app performance.
    
    Returns:
        DataFrame with datetime index, or None if file not found
    """
    # Get project root in a robust way that works from different launch directories
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_path = project_root / "data" / "feature_engineered_events_irregular.csv"
    
    # Check if the engineered data file exists
    if not data_path.exists():
        st.error(f"Engineered data file not found at: {data_path}")
        st.error("Please run the preprocessing pipeline first")
        return None
    
    # Load the data with proper date parsing
    df = pd.read_csv(data_path, index_col='calendar_date', parse_dates=True)
    
    # Sort by date to ensure proper time series order
    return df.sort_index()

def get_prediction(selected_date, df, artifact):
    """
    Generate a DQ_SCORE prediction for the selected date.
    
    This function implements the core prediction logic:
    1. Finds the most recent measurement ≤ selected date (no future data leakage)
    2. Determines which model to use (naive vs ML) based on artifact
    3. Calculates prediction using the appropriate method
    4. Returns prediction, model used, and supporting data
    
    Args:
        selected_date: Date for which to generate forecast
        df: Full dataset with datetime index
        artifact: Model artifact dictionary (may be None for naive only)
        
    Returns:
        Tuple of (prediction, model_used, actual_date, additional_data, features_count)
    """
    # Find the latest measurement date that is ≤ selected date
    # This ensures no future data leakage in predictions
    available_data = df[df.index <= pd.Timestamp(selected_date)]
    
    if available_data.empty:
        return None, None, None, None, 0
    
    # Get the most recent actual measurement date and value
    actual_date = available_data.index[-1]
    # Ensure actual_date is a pandas Timestamp
    if not isinstance(actual_date, pd.Timestamp):
        actual_date = pd.Timestamp(actual_date)
    actual_dq_score = available_data.loc[actual_date, 'DQ_SCORE']
    
    # Determine which model to use based on artifact
    if artifact and artifact.get("selected_model") != "Naive(last_value)":
        # ML MODEL CASE: Use trained machine learning model
        model_used = artifact.get("selected_model")
        
        # Prepare features for ML model prediction
        features = artifact.get("feature_columns", [])  # Fixed: use "feature_columns" not "features"
        features_count = len(features)
        scaler = artifact.get("scaler")
        model = artifact.get("model_object")
        
        # Validate that required features exist in current data
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features[:5]}... (showing first 5)")
            st.error("Model was trained on different data structure. Please retrain the model.")
            # Fallback to Naive
            model_used = "Naive(last_value) (fallback)"
            prediction = actual_dq_score
            additional_data = None
            features_count = 0
            return prediction, model_used, actual_date, additional_data, features_count
        
        # Extract feature values for the prediction date
        X_last = available_data[features].iloc[-1:].values
        
        # Check for empty features or NaN values
        if X_last.shape[1] == 0:
            st.error("No features available for prediction. Falling back to Naive baseline.")
            model_used = "Naive(last_value) (fallback)"
            prediction = actual_dq_score
            additional_data = None
            features_count = 0
            return prediction, model_used, actual_date, additional_data, features_count
        
        # Check for NaN values in features
        if np.isnan(X_last).any():
            st.error("Features contain NaN values. Falling back to Naive baseline.")
            model_used = "Naive(last_value) (fallback)"
            prediction = actual_dq_score
            additional_data = None
            features_count = 0
            return prediction, model_used, actual_date, additional_data, features_count
        
        # Scale features if required (for linear models)
        if scaler is not None:
            try:
                X_last_scaled = scaler.transform(X_last)
                prediction = model.predict(X_last_scaled)[0]
            except Exception as e:
                st.error(f"Error scaling features: {str(e)}. Falling back to Naive baseline.")
                model_used = "Naive(last_value) (fallback)"
                prediction = actual_dq_score
                additional_data = None
                features_count = 0
                return prediction, model_used, actual_date, additional_data, features_count
        else:
            try:
                prediction = model.predict(X_last)[0]
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}. Falling back to Naive baseline.")
                model_used = "Naive(last_value) (fallback)"
                prediction = actual_dq_score
                additional_data = None
                features_count = 0
                return prediction, model_used, actual_date, additional_data, features_count
        
        # Extract feature importances for visualization (tree-based models)
        additional_data = None
        if hasattr(model, 'feature_importances_'):
            # Tree-based model: Feature importances
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)  # Changed from 5 to 10
            additional_data = feature_importance_df
        elif hasattr(model, 'coef_'):
            # Linear model: Coefficients
            coefficients = model.coef_
            # Create DataFrame with features and coefficients
            coef_df = pd.DataFrame({
                'feature': features,
                'coefficient': coefficients,
                'abs_coefficient': abs(coefficients)
            })
            # Filter non-zero coefficients and get top ones
            non_zero_coefs = coef_df[coef_df['abs_coefficient'] > 0]
            top_coefs = non_zero_coefs.nlargest(10, 'abs_coefficient')
            additional_data = {
                'type': 'coefficients',
                'data': top_coefs,
                'features': features,
                'coefficients': coefficients
            }
    
    else:
        # NAIVE BASELINE CASE: Use last observed value
        model_used = "Naive(last_value)"
        prediction = actual_dq_score  # Naive prediction = last value
        additional_data = None
        features_count = 0
    
    return prediction, model_used, actual_date, additional_data, features_count

def get_trend_data(df, current_date):
    """
    Extract the last 10 measurements up to the current date for trend analysis.
    
    Args:
        df: Full dataset with datetime index
        current_date: Date up to which to extract data
        
    Returns:
        DataFrame with measurement numbers, dates, and DQ_SCORE values
    """
    # Filter data up to current date (simulates real-time availability)
    available_data = df[df.index <= current_date].copy()
    
    # Take last 10 measurements for trend visualization
    trend_data = available_data.tail(10)[['DQ_SCORE']].reset_index()
    trend_data.columns = ['date', 'DQ_SCORE']
    trend_data['measurement_num'] = range(1, len(trend_data) + 1)
    
    return trend_data

def plot_coefficient_importance(features, coefficients, top_n=10):
    """
    Create a horizontal bar chart showing top features by coefficient magnitude for linear models.
    Shows only positive coefficients (features that increase DQ_SCORE).
    
    Args:
        features: List of feature names
        coefficients: Array of coefficient values
        top_n: Number of top features to show
        
    Returns:
        matplotlib Figure object
    """
    # Create DataFrame with features and coefficients
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': coefficients
    })
    
    # Filter to only positive coefficients (features that increase DQ_SCORE)
    positive_coefs = coef_df[coef_df['coefficient'] > 0]
    
    # If no positive coefficients, show message
    if positive_coefs.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'No positive coefficients found\nAll features decrease DQ_SCORE', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Positive Feature Coefficients', fontsize=14, fontweight='bold')
        ax.axis('off')
        return fig
    
    # Get top positive coefficients by value
    top_positive = positive_coefs.nlargest(top_n, 'coefficient')
    
    # Sort by coefficient for better visualization
    top_positive = top_positive.sort_values('coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart (all positive, so green color)
    bars = ax.barh(top_positive['feature'], top_positive['coefficient'], 
                   color='#44ff44', alpha=0.8)
    
    # Customize appearance
    ax.set_xlabel('Positive Coefficient Value', fontsize=12)
    ax.set_title(f'Top {len(top_positive)} Features That Increase DQ_SCORE (Positive Coefficients)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add coefficient values as text on bars
    for i, (feature, coef) in enumerate(zip(top_positive['feature'], top_positive['coefficient'])):
        ax.text(coef + 0.001, i, f'+{coef:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_data):
    """
    Create a horizontal bar chart showing top 10 feature importances.
    
    Args:
        feature_data: DataFrame with 'feature' and 'importance' columns
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))  # Increased size for 10 features
    
    # Create horizontal bar chart - better for feature names
    ax.barh(feature_data['feature'], feature_data['importance'], 
            color='#1f77b4', alpha=0.8)
    
    # Customize appearance
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 10 Features Responsible for Prediction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add importance values as text on bars
    for i, (feature, importance) in enumerate(zip(feature_data['feature'], feature_data['importance'])):
        ax.text(importance + 0.01, i, f'{importance:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_trend(trend_df):
    """
    Create a line plot showing DQ_SCORE trend over the last 10 measurements.
    
    Args:
        trend_df: DataFrame with measurement_num, date, and DQ_SCORE columns
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot trend line with markers
    ax.plot(trend_df['measurement_num'], trend_df['DQ_SCORE'], 
            marker='o', linewidth=2, markersize=8, color='#1f77b4')
    
    # Highlight the latest point (used for naive prediction)
    ax.scatter(trend_df['measurement_num'].iloc[-1], trend_df['DQ_SCORE'].iloc[-1], 
               color='#ff7f0e', s=100, zorder=5, label='Latest (Used for Prediction)')
    
    # Customize chart appearance
    ax.set_xlabel('Measurement Number', fontsize=12)
    ax.set_ylabel('DQ_SCORE', fontsize=12)
    ax.set_title('DQ_SCORE Trend (Last 10 Measurements)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotation showing the date and value of the latest measurement
    if len(trend_df) > 0:
        ax.annotate(f"Date: {trend_df['date'].iloc[-1].date()}\nValue: {trend_df['DQ_SCORE'].iloc[-1]:.3f}",
                   xy=(trend_df['measurement_num'].iloc[-1], trend_df['DQ_SCORE'].iloc[-1]),
                   xytext=(trend_df['measurement_num'].iloc[-1]-2, trend_df['DQ_SCORE'].iloc[-1]+0.05),
                   arrowprops=dict(arrowstyle='->', color='gray'),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def plot_full_history_with_numbers(df, current_date):
    """
    Create a comprehensive plot showing ALL DQ_SCORE measurements up to the selected date.
    Uses measurement numbers on x-axis for Naive model visualization.
    
    This function simulates real-time forecasting by only showing data that would
    have been available at the selected date.
    
    Args:
        df: Full dataset with datetime index
        current_date: Selected forecast date
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter data to only include measurements available up to current date
    # This simulates real-time forecasting - no future data leakage
    historical_data = df[df.index <= current_date].copy()
    
    # Create measurement numbers for x-axis
    measurement_numbers = range(1, len(historical_data) + 1)
    
    # Plot line with all measurement points
    ax.plot(measurement_numbers, historical_data['DQ_SCORE'], 
            marker='o', linewidth=2, markersize=6, color='#1f77b4', alpha=0.8)
    
    # Highlight the specific point used for prediction (latest measurement)
    latest_measurement_num = len(historical_data)
    ax.scatter(latest_measurement_num, historical_data['DQ_SCORE'].iloc[-1], 
               color='#ff7f0e', s=120, zorder=5, label='Used for Prediction', edgecolor='black')
    
    # Customize chart appearance
    ax.set_xlabel('Measurement Number', fontsize=12)
    ax.set_ylabel('DQ_SCORE', fontsize=12)
    ax.set_title('DQ_SCORE History Up To Selected Date (All Measurements)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add value labels on points if dataset is manageable (≤30 points)
    if len(historical_data) <= 30:
        for i, (num, score) in enumerate(zip(measurement_numbers, historical_data['DQ_SCORE'])):
            ax.annotate(f'{score:.2f}', 
                       (num, score), 
                       xytext=(0, 5), 
                       textcoords='offset points',
                       ha='center', 
                       fontsize=8,
                       alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_full_history(df, current_date):
    """
    Create a comprehensive plot showing ALL DQ_SCORE measurements up to the selected date.
    Uses actual dates on x-axis for better temporal understanding.
    
    This function simulates real-time forecasting by only showing data that would
    have been available at the selected date.
    
    Args:
        df: Full dataset with datetime index
        current_date: Selected forecast date
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter data to only include measurements available up to current date
    # This simulates real-time forecasting - no future data leakage
    historical_data = df[df.index <= current_date].copy()
    
    # Plot line with all measurement points
    ax.plot(historical_data.index, historical_data['DQ_SCORE'], 
            marker='o', linewidth=2, markersize=6, color='#1f77b4', alpha=0.8)
    
    # Highlight the specific point used for prediction (latest measurement)
    ax.scatter(current_date, df.loc[current_date, 'DQ_SCORE'], 
               color='#ff7f0e', s=120, zorder=5, label='Used for Prediction', edgecolor='black')
    
    # Customize chart appearance
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('DQ_SCORE', fontsize=12)
    ax.set_title('DQ_SCORE History Up To Selected Date', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis to display dates nicely (auto-rotate for readability)
    fig.autofmt_xdate()
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on points if dataset is manageable (≤30 points)
    # This prevents overcrowding on larger datasets
    if len(historical_data) <= 30:
        for date, score in historical_data['DQ_SCORE'].items():
            ax.annotate(f'{score:.2f}', 
                       (date, score), 
                       xytext=(0, 5), 
                       textcoords='offset points',
                       ha='center', 
                       fontsize=8,
                       alpha=0.7)
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main Streamlit application function.
    
    This function orchestrates the entire forecasting application:
    1. Loads data and model artifacts
    2. Sets up the UI layout and sidebar
    3. Handles user interactions (date selection, prediction requests)
    4. Displays results with appropriate visualizations
    5. Provides options for additional details and historical context
    """
    # -------------------------------------------------------------------------
    # DATA LOADING AND VALIDATION
    # -------------------------------------------------------------------------
    
    # Load the engineered dataset and latest model artifact
    df = load_engineered_data()
    artifact = load_forecast_artifact()
    
    # Validate data loading - stop app if data is unavailable
    if df is None:
        st.error("Failed to load data. Please check the data file exists.")
        st.stop()
    
    # Remove debug lines for production
    # st.write("Data loaded successfully!")  # Debug info
    # st.write(f"Data shape: {df.shape}")  # Debug info
    
    # -------------------------------------------------------------------------
    # PAGE HEADER AND SIDEBAR SETUP
    # -------------------------------------------------------------------------
    
    # Main application title
    st.title("📊 DQ_SCORE Forecast")
    st.markdown("---")
    
    # Sidebar configuration - shows model information and data summary
    with st.sidebar:
        st.subheader("Model Information")
        
        # Display which model is being used (ML vs Naive)
        if artifact and artifact.get("selected_model") != "Naive(last_value)":
            st.success(f"🤖 **ML Model**: {artifact.get('selected_model')}")
            st.info("Using trained machine learning model for prediction")
        else:
            st.info("📈 **Naive Baseline**: Last Value Method")
            st.info("Prediction based on most recent DQ_SCORE value")
        
        st.markdown("---")
        st.subheader("Data Summary")
        st.metric("Total Measurements", len(df))
        st.write(f"**Latest Measurement:** {df.index[-1].date()}")
        st.metric("Latest DQ_SCORE", f"{df['DQ_SCORE'].iloc[-1]:.3f}")
        
        # Show model performance metrics if available
        if artifact and 'metrics_summary' in artifact:
            st.markdown("---")
            st.subheader("📊 Model Performance (RMSE)")
            st.dataframe(artifact['metrics_summary'])
    
    # -------------------------------------------------------------------------
    # MAIN CONTENT LAYOUT
    # -------------------------------------------------------------------------
    
    # Create two-column layout: controls on left, results on right
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📅 Forecast Configuration")
        
        # -----------------------------------------------------------------
        # DATE INPUT CONTROLS
        # -----------------------------------------------------------------
        
        # Set valid date range for forecasting
        min_date = df.index[0].date()  # Earliest measurement date
        max_date = df.index[-1].date() + pd.Timedelta(days=365)  # Allow 1 year into future
        
        # Date input widget with validation
        selected_date = st.date_input(
            "Select forecast date",
            value=df.index[-1].date(),  # Default to latest measurement
            min_value=min_date,
            max_value=max_date
        )
        
        # -----------------------------------------------------------------
        # PREDICTION TRIGGER AND OPTIONS
        # -----------------------------------------------------------------
        
        # Main prediction button - triggers the forecasting process
        predict_button = st.button("🔮 Get Prediction", type="primary", use_container_width=True)
        
        # Advanced options for additional insights
        with st.expander("Advanced Options"):
            show_details = st.checkbox("Show technical details", value=False)
            show_history = st.checkbox("Show historical trend", value=True)
    
    # -------------------------------------------------------------------------
    # RESULTS DISPLAY COLUMN
    # -------------------------------------------------------------------------
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        # -----------------------------------------------------------------
        # PREDICTION GENERATION AND DISPLAY
        # -----------------------------------------------------------------
        
        if predict_button:
            # Show loading spinner while processing
            with st.spinner("Generating prediction..."):
                # Generate prediction using the selected date
                prediction, model_used, actual_date, additional_data, features_count = get_prediction(selected_date, df, artifact)
                
                # Handle prediction errors
                if prediction is None:
                    st.error("Unable to generate prediction. Please try a different date.")
                else:
                    # -----------------------------------------------------------------
                    # PREDICTION RESULTS DISPLAY
                    # -----------------------------------------------------------------
                    
                    # Display the main prediction result with custom styling
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric(
                        "Predicted Next DQ_SCORE",
                        f"{prediction:.3f}",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show which historical measurement was used for prediction
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    # Handle both datetime and string cases for actual_date
                    if hasattr(actual_date, 'date'):
                        date_display = actual_date.date()
                    else:
                        date_display = actual_date
                    st.info(f"📌 Based on measurement from: **{date_display}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display the model type used for this prediction
                    if model_used == "Naive(last_value)":
                        st.success("📈 Using Naive Baseline (Last Value Method)")
                    else:
                        st.success(f"🤖 Using ML Model: {model_used}")
                    
                    # -----------------------------------------------------------------
                    # VISUALIZATION SECTION
                    # -----------------------------------------------------------------
                    
                    st.markdown("---")
                    st.subheader("📈 Analysis")
                    
                    # -----------------------------------------------------------------
                    # MODEL-SPECIFIC VISUALIZATIONS
                    # -----------------------------------------------------------------
                    
                    if model_used == "Naive(last_value)":
                        # NAIVE MODEL: Show trend analysis
                        st.write("**DQ_SCORE Trend (Naive uses the last value as prediction):**")
                        # Generate trend data for visualization
                        trend_data = get_trend_data(df, actual_date)
                        fig = plot_trend(trend_data)
                        st.pyplot(fig)
                        
                        # Show technical details if requested
                        if show_details:
                            st.markdown("**How Naive Works:**")
                            st.code("prediction = last_observed_DQ_SCORE")
                            # Handle both datetime and string cases for actual_date
                            if hasattr(actual_date, 'date'):
                                date_display = actual_date.date()
                            else:
                                date_display = actual_date
                            st.write(f"Latest DQ_SCORE ({date_display}): {df.loc[actual_date, 'DQ_SCORE']:.3f}")
                            st.write(f"Prediction: {prediction:.3f}")
                    
                    elif model_used != "Naive(last_value)":
                        # ML MODEL: Show feature importance analysis or coefficient analysis
                        if additional_data is not None:
                            if isinstance(additional_data, dict) and additional_data.get('type') == 'coefficients':
                                # Linear model (ElasticNet, Ridge, etc.): Show positive coefficient importance
                                st.write("**Top Features That Increase DQ_SCORE (Positive Coefficients):**")
                                fig = plot_coefficient_importance(
                                    additional_data['features'], 
                                    additional_data['coefficients']
                                )
                                st.pyplot(fig)
                                
                                # Show technical details if requested
                                if show_details:
                                    st.markdown("**Positive Coefficient Details:**")
                                    coef_data = additional_data['data']
                                    positive_coefs = coef_data[coef_data['coefficient'] > 0]
                                    if positive_coefs.empty:
                                        st.info("No features with positive coefficients found. All selected features decrease DQ_SCORE.")
                                    else:
                                        for _, row in positive_coefs.iterrows():
                                            feature_val = df.loc[actual_date, row['feature']]
                                            st.write(f"• **{row['feature']}**: {feature_val:.3f} (coef: +{row['coefficient']:.4f})")
                            else:
                                # Tree-based model: Show feature importance
                                st.write("**Top 10 Features Responsible for This Prediction:**")
                                fig = plot_feature_importance(additional_data)
                                st.pyplot(fig)
                                
                                # Show technical details if requested
                                if show_details:
                                    st.markdown("**Feature Details:**")
                                    for _, row in additional_data.iterrows():
                                        feature_val = df.loc[actual_date, row['feature']]
                                        st.write(f"• **{row['feature']}**: {feature_val:.3f} (importance: {row['importance']:.3f})")
                        else:
                            # Fallback: Show trend analysis
                            st.write(f"**DQ_SCORE Trend ({model_used} Analysis):**")
                            trend_data = get_trend_data(df, actual_date)
                            fig = plot_trend(trend_data)
                            st.pyplot(fig)
                            
                            # Show technical details if requested
                            if show_details:
                                st.markdown(f"**How {model_used} Works:**")
                                st.code(f"prediction = model.predict(scaled_features)")
                                # Handle both datetime and string cases for actual_date
                                if hasattr(actual_date, 'date'):
                                    date_display = actual_date.date()
                                else:
                                    date_display = actual_date
                                st.write(f"Latest DQ_SCORE ({date_display}): {df.loc[actual_date, 'DQ_SCORE']:.3f}")
                                st.write(f"Prediction: {prediction:.3f}")
                                st.write(f"Features used: {features_count} engineered features")
                                st.write(f"Scaling: Applied (StandardScaler)")
                    
                    # -----------------------------------------------------------------
                    # HISTORICAL CONTEXT SECTION (OPTIONAL)
                    # -----------------------------------------------------------------
                    
                    # Show additional historical trend if user requested it
                    if show_history:
                        st.markdown("---")
                        st.subheader("📊 Historical DQ_SCORE")
                        
                        if model_used == "Naive(last_value)":
                            # NAIVE MODEL: Show all measurements up to selected date with measurement numbers
                            fig_hist = plot_full_history_with_numbers(df, actual_date)
                            st.pyplot(fig_hist)
                        else:
                            # ML MODEL: Show complete history with actual dates
                            fig_hist = plot_full_history(df, actual_date)
                            st.pyplot(fig_hist)
        else:
            # Show initial prompt when no prediction has been made yet
            st.info("👈 Select a date and click 'Get Prediction' to see the forecast")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the main Streamlit application
    main()
