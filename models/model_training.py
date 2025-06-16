"""
Model training module for fraud detection.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

from utils.data_preprocessing import preprocess_data, balance_dataset
from utils.feature_engineering import engineer_features
from config.config import (
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    CV_FOLDS,
    LR_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    IF_PARAMS
)


def load_and_prepare_data(data_file: str) -> tuple:
    """
    Load and prepare data for model training.
    
    Args:
        data_file: Path to the data file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, encoder)
    """
    # Load data
    df = pd.read_csv(data_file)
    
    # Engineer features
    df = engineer_features(df)
    
    # Preprocess data
    X_processed, y, scaler, encoder = preprocess_data(df, training=True)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )
    
    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(
        X_train, 
        y_train, 
        random_state=RANDOM_SEED
    )
    
    return X_train_balanced, X_test, y_train_balanced, y_test, scaler, encoder


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        LogisticRegression: Trained model
    """
    print("Training Logistic Regression model...")
    
    # Create model with hyperparameters
    model = LogisticRegression(**LR_PARAMS)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a random forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("Training Random Forest model...")
    
    # Create model with hyperparameters
    model = RandomForestClassifier(**RF_PARAMS)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        XGBClassifier: Trained model
    """
    print("Training XGBoost model...")
    
    # Create model with hyperparameters
    model = XGBClassifier(**XGB_PARAMS)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def train_isolation_forest(X_train: pd.DataFrame) -> IsolationForest:
    """
    Train an isolation forest model for anomaly detection.
    
    Args:
        X_train: Training features
        
    Returns:
        IsolationForest: Trained model
    """
    print("Training Isolation Forest model...")
    
    # Create model with hyperparameters
    model = IsolationForest(**IF_PARAMS)
    
    # Train model
    model.fit(X_train)
    
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    is_anomaly_model: bool = False
) -> dict:
    """
    Evaluate a trained model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        is_anomaly_model: Whether the model is an anomaly detection model
        
    Returns:
        dict: Model evaluation metrics
    """
    print(f"Evaluating {model_name}...")
    
    if is_anomaly_model:
        # For anomaly detection models like Isolation Forest
        # -1 for anomalies (frauds), 1 for normal
        raw_scores = model.decision_function(X_test)
        # Convert to probability-like score (0 to 1)
        # Lower decision_function values mean more anomalous
        scores = 1 - (raw_scores + 0.5)  # Normalize from [-0.5, 0.5] to [1, 0]
        scores = np.clip(scores, 0, 1)  # Clamp between 0 and 1
        
        y_pred = model.predict(X_test)
        # Convert from -1/1 to 0/1 prediction
        y_pred = [1 if p == -1 else 0 for p in y_pred]
    else:
        # For classification models
        y_pred = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, scores)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print evaluation
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")
    
    # Return metrics
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def save_model(model, file_name: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Model to save
        file_name: File name for the saved model
    """
    # Ensure directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Save model
    with open(os.path.join(MODEL_PATH, file_name), 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {os.path.join(MODEL_PATH, file_name)}")


def train_all_models(data_file: str) -> dict:
    """
    Train and evaluate all models.
    
    Args:
        data_file: Path to the data file
        
    Returns:
        dict: Dictionary containing all trained models and their metrics
    """
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_prepare_data(data_file)
    
    # Dictionary to store models and their metrics
    results = {}
    
    # Train and evaluate logistic regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    save_model(lr_model, "logistic_regression_model.pkl")
    results['logistic_regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # Train and evaluate random forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    save_model(rf_model, "random_forest_model.pkl")
    results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # Train and evaluate XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    save_model(xgb_model, "xgboost_model.pkl")
    results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # Train and evaluate isolation forest
    if_model = train_isolation_forest(X_train)
    if_metrics = evaluate_model(if_model, X_test, y_test, "Isolation Forest", is_anomaly_model=True)
    save_model(if_model, "isolation_forest_model.pkl")
    results['isolation_forest'] = {'model': if_model, 'metrics': if_metrics}
    
    # Save preprocessors
    save_model(scaler, "scaler.pkl")
    save_model(encoder, "encoder.pkl")
    
    return results


if __name__ == "__main__":
    data_file = os.path.join("data", "sample_data.csv")
    train_all_models(data_file) 