# Capstone Project: ML Data Quality Score Analysis

## Project Overview
This project analyzes data quality test results from a comprehensive healthcare data validation system. The goal is to build machine learning models that can predict data quality scores and identify potential data issues before they impact downstream systems. The system processes raw test events from multiple data quality categories including allocation, completeness, uniqueness, referential integrity, privacy, and schema validation.

## Dataset Description

### Initial Dataset: `pseudo_deident.csv`
The raw dataset contains 16 columns of pseudo-anonymized data quality test results:

| Column Name | Description |
|-------------|-------------|
| `TEST_RUN_ID` | Unique identifier for test execution runs (e.g., RUN_421) |
| `NAME_OF_TEST` | Type of data quality test (DUPLICATES, REFERENTIAL_INTEGRITY, ALLOCATION, etc.) |
| `SOURCE_DATABASE` | Source database name where the test was executed |
| `SOURCE_SCHEMA` | Source schema name within the database |
| `SOURCE_TABLE` | Source table name being tested |
| `SOURCE_COLUMN` | Source column name being tested (may be empty) |
| `TARGET_DATABASE` | Target database name for comparison tests |
| `TARGET_SCHEMA` | Target schema name for comparison tests |
| `TARGET_TABLE` | Target table name for comparison tests |
| `TARGET_COLUMN` | Target column name for comparison tests (may be empty) |
| `TEST_RESULT` | Number of issues found (0 for pass, >0 for failures) |
| `STATUS` | Test outcome (PASS/FAIL) |
| `DETAILS` | Human-readable description of test results |
| `SEVERITY_LEVEL` | Impact severity of the test (MEDIUM SEVERITY, HIGH SEVERITY) |
| `RUN_TIMESTAMP` | When the test was executed (ISO format) |
| `IS_LATEST` | Flag for most recent test results (0/1) |

### Processed Dataset: `events.csv`
The enhanced dataset contains 23 columns with additional engineered features including DQ_EVENT (JSON structure), TRUST_SCORE, and temporal aggregations.

## Why Data Quality Tests Matter in Data Engineering

### **🏗️ Foundation of Reliable Data Systems**
Data quality tests are the **cornerstone of trustworthy data engineering** because they ensure that data pipelines deliver accurate, consistent, and reliable information to downstream systems and decision-makers.

### **🔍 Key Data Engineering Perspectives:**

#### **1. Pipeline Reliability Assurance**
- **Early Detection**: Tests catch issues before they propagate through complex data pipelines
- **Automated Validation**: Continuous monitoring prevents silent data corruption
- **Pipeline Health**: Test results serve as vital signs for data infrastructure performance

#### **2. Data Trust & Governance**
- **Stakeholder Confidence**: Test results provide evidence of data reliability for business users
- **Compliance Requirements**: Regulatory frameworks (HIPAA, GDPR) mandate data quality validation
- **Audit Trails**: Test execution logs create verifiable records of data quality practices

#### **3. Cost Prevention & Risk Mitigation**
- **Downstream Error Prevention**: Bad data causes costly errors in analytics, reporting, and decision-making
- **Operational Efficiency**: Automated tests reduce manual data validation efforts
- **Risk Management**: Quality metrics identify potential data-related business risks

#### **4. System Integration Validation**
- **Cross-System Consistency**: Tests verify data integrity across databases, schemas, and applications
- **Schema Evolution**: Tests detect breaking changes during system updates or migrations
- **Data Flow Validation**: Ensures data transformations preserve intended meaning and structure

#### **5. Performance & Scalability Indicators**
- **System Load Monitoring**: Test execution times and resource usage indicate system performance
- **Complexity Metrics**: Advanced test patterns reveal system sophistication and maturity
- **Capacity Planning**: Test results help forecast infrastructure needs for data growth

### **🎯 Healthcare Data Engineering Specifics**

#### **Patient Safety & Clinical Decision Support**
- **Life-Critical Accuracy**: Medical data quality directly impacts patient care and safety
- **Clinical Validity**: Tests ensure medical data follows healthcare standards and terminologies
- **Care Coordination**: Quality tests validate data consistency across healthcare systems

#### **Regulatory Compliance & Privacy**
- **HIPAA Validation**: Tests ensure PHI/PII data is properly handled and protected
- **Audit Requirements**: Healthcare regulations demand documented data quality practices
- **Privacy Preservation**: Tests verify data anonymization and de-identification processes

#### **Interoperability Standards**
- **FHIR/HL7 Compliance**: Tests validate healthcare data exchange standards
- **Cross-Institution Data**: Quality checks ensure data consistency between healthcare providers
- **Semantic Integrity**: Tests preserve medical meaning across system boundaries

### **📊 From Data Engineering to Machine Learning**

This project transforms traditional data quality testing into **predictive intelligence**:

- **Reactive Testing** → **Proactive Forecasting**
- **Manual Analysis** → **Automated Insights**
- **Historical Reporting** → **Future Planning**
- **Quality Monitoring** → **Quality Intelligence**

### **🚀 Business Impact of Quality Testing**

- **Decision Confidence**: High-quality data enables reliable business intelligence and analytics
- **Operational Excellence**: Quality tests prevent costly data-related operational issues
- **Customer Trust**: Consistent data quality builds trust in products and services
- **Innovation Enablement**: Reliable data foundation supports advanced analytics and AI initiatives

**Bottom Line:** Data quality tests are **not just technical checks**—they are **fundamental business safeguards** that enable data-driven organizations to operate with confidence, compliance, and competitive advantage.

## Project Structure
```
Capstone_Project_ML_DQ_SCORE/
├── data/
│   ├── pseudo_deident.csv          # Raw pseudo-anonymized dataset (16 columns)
│   ├── events.csv                  # Enhanced dataset with DQ_EVENT (23 columns)
│   ├── feature_engineered_events.csv           # Standard engineered features (286 columns)
│   └── feature_engineered_events_irregular.csv  # Irregular time series features (250 columns)
├── notebooks/
│   └── Capstone_notebook.ipynb                 # Complete workflow notebook including EDA of the data
├── src/
│   ├── enhanced_preprocessing_pipeline.py       # Main pipeline orchestrator
│   ├── preprocessing_utils.py                   # Pipeline utilities and data loading
│   ├── feature_engineer.py                      # Feature engineering implementation
│   ├── irregular_time_series_processor.py       # Irregular time series handling
│   ├── auto_forecast.py                         # Automated model selection and training
│   ├── streamlit_app.py                         # Interactive forecasting dashboard
│   ├── dq_parsing/                              # Test event parsing modules
│   │   ├── allocation.py                        # Allocation test parsing
│   │   ├── completeness.py                      # Completeness test parsing
│   │   ├── uniqueness.py                        # Uniqueness test parsing
│   │   ├── referential_integrity.py             # Referential integrity parsing
│   │   ├── privacy.py                           # Privacy test parsing
│   │   ├── schema.py                            # Schema test parsing
│   │   ├── person.py                            # Person test parsing
│   │   └── utils.py                             # Parsing utilities
│   └── [additional modules...]                   # Other supporting modules
├── models/
│   ├── dq_score_next_forecaster_*.pkl           # Trained forecasting models
│   └── dq_score_multi_model_pipeline.pkl         # Multi-model pipeline artifacts
├── analysis.md                                   # Comprehensive analysis documentation
├── capstone_decisions.md                         # Technical decisions and rationale
├── requirements.txt                              # Python dependencies
└── README.md                                     # Project documentation
```

## Installation and Setup

### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv dq_ml_env

# Activate virtual environment
# Windows:
dq_ml_env\Scripts\activate
# Mac/Linux:
source dq_ml_env/bin/activate
```

### 2. Clone Repository
```bash
git clone https://github.com/hermitquant/Capstone_Project_ML_DQ_SCORE.git
cd Capstone_Project_ML_DQ_SCORE
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

### 5. Open the Workflow Notebook
Navigate to `notebooks/Capstone_notebook.ipynb` to run the complete analysis pipeline.

## Capstone Notebook Workflow

The `Capstone_notebook.ipynb` contains the complete end-to-end workflow:

### Step 1: Environment Setup
- Install all required dependencies
- Import necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Configure project paths and system settings

### Step 2: Data Processing Pipeline
- **Build `events.csv` from `pseudo_deident.csv`**: Transform raw test events into enhanced dataset with DQ_EVENT JSON structures
- **Generate engineered datasets**: Create both standard and irregular time series feature sets
- **Run preprocessing pipeline**: Execute complete feature engineering with validation

### Step 3: Model Training and Selection
- **Automatic walk-forward model selection**: Test multiple algorithms (ElasticNet, DecisionTree, BayesianRidge, Ridge)
- **Cross-validation with time series splits**: Ensure temporal validation without data leakage
- **Model comparison**: Select best performing model based on MSE and R² metrics
- **Save model artifacts**: Serialize trained models with timestamped filenames

### Step 4: Forecasting and Analysis
- **Next measurement prediction**: Forecast DQ_SCORE for upcoming time periods
- **Feature importance analysis**: Identify key drivers of data quality scores
- **Model interpretation**: Understand coefficient importance and decision tree logic

### Step 5: Visualization and Reporting
- **Interactive dashboard**: Launch Streamlit app for real-time forecasting
- **Performance metrics**: Display model accuracy and confidence intervals
- **Feature importance charts**: Visualize top predictive features

## Key Features and Capabilities

### Post-Run vs Pre-Run Forecasting Modes
- **Post-Run Mode**: Analysis mode with complete daily data (85-90% accuracy)
- **Pre-Run Mode**: True forecasting mode with limited information (70-80% accuracy)
- **Mode-dependent model selection**: ElasticNet for post-run, DecisionTree for pre-run

### Automated Model Selection
- **ElasticNet**: Linear model with regularization (selected for rich feature sets)
- **DecisionTree**: Non-linear model (selected for limited feature scenarios)
- **BayesianRidge**: Probabilistic linear regression
- **Ridge**: Regularized linear regression

### Feature Engineering Pipeline
- **286 standard features**: Comprehensive temporal and categorical features
- **250 irregular features**: Gap-aware time series features for irregular data
- **JSON parsing**: Extract structured data from DQ_EVENT fields
- **Temporal aggregations**: Daily, weekly, and monthly summary statistics

## Tools and Technologies
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Jupyter Notebooks**: Interactive analysis environment
- **Joblib**: Model serialization and parallel processing

## Key Insights from Analysis

### Top Predictive Features (Post-Run Mode)
1. **daily_metric_alloc_diff_mean** (+0.007128): Resource allocation changes
2. **daily_complexity_score_max** (+0.004493): System complexity handling
3. **daily_metric_count_person_id_sum** (+0.004364): Person-based coverage
4. **daily_metric_count_distinct_person_id_sum** (+0.004333): Engagement diversity

### Key Predictive Features (Pre-Run Mode)
1. **passed_tests** (60% importance): Test execution volume
2. **measurements_last_30d** (40% importance): Recent activity consistency

### Business Implications
- **Resource allocation optimization** has strongest impact on data quality
- **System complexity handling** indicates mature data infrastructure
- **Person-based metrics** reflect comprehensive data coverage
- **Test volume and consistency** are key leading indicators

## Next Steps and Future Development
1. **Real-time monitoring**: Deploy automated monitoring for production data quality
2. **Anomaly detection**: Implement unsupervised learning for unusual pattern detection
3. **Recommendation system**: Develop actionable insights for data quality improvements
4. **Multi-tenant support**: Extend system for multiple organizational deployments
5. **Advanced forecasting**: Incorporate external factors and seasonality patterns

## Contributing
This project represents a comprehensive approach to healthcare data quality management using machine learning. The modular architecture allows for easy extension and customization based on specific organizational needs.

## License
This project is part of a capstone demonstration and is provided for educational and research purposes.
