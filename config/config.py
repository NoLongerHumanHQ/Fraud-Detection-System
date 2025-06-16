"""
Configuration settings for the Fraud Detection System.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_data.csv")

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, "saved_models")

# Model training configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model hyperparameters
# Logistic Regression
LR_PARAMS = {
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': RANDOM_SEED
}

# Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 10,  # For class imbalance
    'random_state': RANDOM_SEED
}

# Isolation Forest
IF_PARAMS = {
    'n_estimators': 100,
    'contamination': 'auto',
    'max_samples': 'auto',
    'random_state': RANDOM_SEED
}

# Feature engineering settings
TIME_FEATURES = ['hour_of_day', 'day_of_week', 'month', 'is_weekend']

# Streamlit settings
THEME_COLOR = "#0068C9"
MAX_UPLOAD_SIZE = 200  # MB 