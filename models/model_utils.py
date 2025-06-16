"""
Model utilities for fraud detection.
"""
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sklearn.base import BaseEstimator

from utils.data_preprocessing import preprocess_data
from utils.feature_engineering import engineer_features
from config.config import MODEL_PATH


def load_model(model_file: str) -> Any:
    """
    Load a saved model from disk.
    
    Args:
        model_file: Name of the model file
        
    Returns:
        Any: Loaded model
    """
    file_path = os.path.join(MODEL_PATH, model_file)
    
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def load_all_models() -> Dict[str, Any]:
    """
    Load all saved models and preprocessors.
    
    Returns:
        Dict[str, Any]: Dictionary of loaded models and preprocessors
    """
    model_files = {
        'logistic_regression': 'logistic_regression_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'isolation_forest': 'isolation_forest_model.pkl',
        'scaler': 'scaler.pkl',
        'encoder': 'encoder.pkl'
    }
    
    models = {}
    
    for model_name, file_name in model_files.items():
        try:
            models[model_name] = load_model(file_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name} model: {e}")
    
    return models


def predict_fraud(
    transaction_data: Dict[str, Any],
    models: Dict[str, Any],
    model_name: str = 'random_forest'
) -> Tuple[float, int, str]:
    """
    Predict fraud probability for a single transaction.
    
    Args:
        transaction_data: Dictionary of transaction data
        models: Dictionary of loaded models
        model_name: Name of the model to use for prediction
        
    Returns:
        Tuple[float, int, str]: (fraud_probability, fraud_score, risk_category)
    """
    # Convert transaction data to dataframe
    df = pd.DataFrame([transaction_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Preprocess data
    df_processed = preprocess_data(df, models['scaler'], models['encoder'], training=False)
    
    # Make prediction
    model = models[model_name]
    
    if model_name == 'isolation_forest':
        # For anomaly detection models
        raw_score = model.decision_function([df_processed.iloc[0].values])[0]
        # Convert to probability-like score (0 to 1)
        probability = 1 - (raw_score + 0.5)  # Normalize from [-0.5, 0.5] to [1, 0]
        probability = max(0, min(probability, 1))  # Clamp between 0 and 1
    else:
        # For classification models
        probability = model.predict_proba([df_processed.iloc[0].values])[0][1]
    
    # Calculate fraud score (0-100)
    fraud_score = int(probability * 100)
    
    # Determine risk category
    if fraud_score < 30:
        risk_category = "Low"
    elif fraud_score < 70:
        risk_category = "Medium"
    else:
        risk_category = "High"
    
    return probability, fraud_score, risk_category


def predict_batch_fraud(
    batch_data: pd.DataFrame,
    models: Dict[str, Any],
    model_name: str = 'random_forest'
) -> pd.DataFrame:
    """
    Predict fraud for multiple transactions.
    
    Args:
        batch_data: DataFrame of multiple transactions
        models: Dictionary of loaded models
        model_name: Name of the model to use for prediction
        
    Returns:
        pd.DataFrame: Original data with fraud predictions
    """
    # Engineer features
    df = engineer_features(batch_data)
    
    # Preprocess data
    df_processed = preprocess_data(df, models['scaler'], models['encoder'], training=False)
    
    # Make predictions
    model = models[model_name]
    
    if model_name == 'isolation_forest':
        # For anomaly detection models
        raw_scores = model.decision_function(df_processed.values)
        # Convert to probability-like scores (0 to 1)
        probabilities = 1 - (raw_scores + 0.5)  # Normalize from [-0.5, 0.5] to [1, 0]
        probabilities = np.clip(probabilities, 0, 1)  # Clamp between 0 and 1
    else:
        # For classification models
        probabilities = model.predict_proba(df_processed.values)[:, 1]
    
    # Add predictions to original data
    results = batch_data.copy()
    results['fraud_probability'] = probabilities
    results['fraud_score'] = results['fraud_probability'].apply(lambda x: int(x * 100))
    results['risk_category'] = results['fraud_score'].apply(
        lambda x: "Low" if x < 30 else ("Medium" if x < 70 else "High")
    )
    
    return results


def get_model_performance(model_name: str) -> Dict[str, Union[float, str, np.ndarray]]:
    """
    Get stored performance metrics for a model.
    
    In a real implementation, these would be loaded from saved model metrics.
    This is a placeholder implementation with hardcoded values for demo purposes.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dict[str, Union[float, str, np.ndarray]]: Dictionary of performance metrics
    """
    # Sample metrics for demonstration purposes
    # In a real implementation, these would be loaded from saved evaluation results
    metrics = {
        'logistic_regression': {
            'accuracy': 0.92,
            'precision': 0.86,
            'recall': 0.79,
            'f1': 0.82,
            'auc': 0.91,
            'confusion_matrix': np.array([[955, 45], [21, 79]])
        },
        'random_forest': {
            'accuracy': 0.96,
            'precision': 0.91,
            'recall': 0.85,
            'f1': 0.88,
            'auc': 0.95,
            'confusion_matrix': np.array([[970, 30], [15, 85]])
        },
        'xgboost': {
            'accuracy': 0.97,
            'precision': 0.92,
            'recall': 0.86,
            'f1': 0.89,
            'auc': 0.96,
            'confusion_matrix': np.array([[975, 25], [14, 86]])
        },
        'isolation_forest': {
            'accuracy': 0.91,
            'precision': 0.84,
            'recall': 0.76,
            'f1': 0.80,
            'auc': 0.88,
            'confusion_matrix': np.array([[950, 50], [24, 76]])
        }
    }
    
    return metrics.get(model_name, {}) 