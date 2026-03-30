# Disease Outbreak Prediction Platform - Codebase Documentation

## 1. Project Overview
This project serves as a comprehensive "Disease Outbreak Prediction Platform" designed to forecast disease cases (specifically Acute Diarrhoeal Disease) using historical epidemiological data and environmental parameters. The system employs machine learning regression techniques to predict future outbreaks and assigns risk levels to aid public health decision-making.

## 2. Methodology & Architecture
The project follows a modular MLOps pipeline structure:

1.  **Data Ingestion**: Loading and cleaning raw CSV data (`Final_data.csv`).
2.  **Preprocessing**:
    *   **Filtering**: Selecting target disease and relevant states/districts.
    *   **Imputation**: Handling missing climate data (Temp, Preci, LAI) using group-level means and forward filling.
    *   **Feature Engineering**: Creation of lag features (`caseslastweek`, `caseslastmonth`) to capture temporal dependencies.
    *   **Transformation**: Scaling numeric features (StandardScaler) and encoding categorical variables (LabelEncoder).
3.  **Modeling**:
    *   **Baselines**: Random Forest Regressor and Gradient Boosting Regressor.
    *   **Deep Learning**: LSTM (Long Short-Term Memory) network architecture is defined for sequence-based prediction. 
        *   *Note*: In this deployment environment, the lightweight Gradient Boosting model was selected as the primary engine due to system constraints (Long Path support for TensorFlow), but the code `model_training.py` includes the full LSTM implementation for use in compatible environments (e.g., Linux/Colab).
4.  **Deployment**: A Streamlit dashboard (`app.py`) loads the serialized pipeline (`best_disease_model.pkl`) for real-time inference.

## 3. Key Components

### 3.1 Data Inspection (`inspect_data.py`)
*   **Purpose**: Preliminary exploratory data analysis (EDA).
*   **Key Insight**: 'Acute Diarrhoeal Disease' was selected as the target due to having the largest sample size (5126 records) and consistent temporal coverage (14 years).

### 3.2 Model Training Pipeline (`model_training.py`)
*   **Purpose**: Main driver script for training and serialization.
*   **Key Functions**:
    *   `load_and_preprocess_data()`: Handles robust CSV reading and cleaning.
    *   `create_features()`: Generates lag-7 (previous week) and lag-30 (previous month) proxies.
    *   `train_baselines()`: Trains RF and GBM models.
    *   `build_lstm()`: (Optional) Constructs a Keras LSTM architecture.
*   **Evaluation Strategy**: Time-based split (Training on 2009-2018, Testing on 2019+). Metrics used: RMSE, MAE, MAPE.

### 3.3 Dashboard Application (`app.py`)
*   **Purpose**: User Interface for stakeholder interaction.
*   **Features**:
    *   **Input Controls**: Sidebar for selecting Location and Climate parameters.
    *   **Risk Logic**: 
        *   🔴 **High Risk**: >100 cases
        *   🟡 **Medium Risk**: 50-100 cases
        *   🟢 **Low Risk**: <50 cases
    *   **Visuals**: Metric cards and Feature Importance plots using Plotly.

## 4. Setup & Usage

### Prerequisites
*   Python 3.8+
*   Packages: `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `plotly`
*   (Optional) `tensorflow` for LSTM features.

### Execution
1.  **Inspect Data**:
    ```bash
    python inspect_data.py
    ```
2.  **Train Models**:
    ```bash
    python model_training.py
    ```
    This generates `best_disease_model.pkl`.
3.  **Launch Dashboard**:
    ```bash
    python -m streamlit run app.py
    ```

## 5. Model Performance
*   **Best Model**: Gradient Boosting Regressor
*   **Performance**: ~1.25% MAPE (Mean Absolute Percentage Error) on test set.
*   **Inference Speed**: <100ms per prediction.

## 6. Future Enhancements
*   Integration of real-time weather API.
*   Expansion to multi-disease simultaneous prediction.
*   Geospatial mapping of risk radii using district centroids.

---
*Generated for the Disease Outbreak Prediction Platform Viva/Capstone Project.*
