"""
Data preprocessing utilities for the fraud detection system.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def identify_features(df: pd.DataFrame) -> tuple:
    """
    Identify numerical and categorical features from the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        tuple: (numerical_features, categorical_features)
    """
    # Exclude target variable and ID columns if present
    exclude_cols = ['is_fraud', 'transaction_id', 'user_id', 'timestamp']
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Identify numerical and categorical features
    numerical_features = []
    categorical_features = []
    
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    return numerical_features, categorical_features


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    numerical_features, categorical_features = identify_features(df)
    
    # Create a copy to avoid modifying the original dataframe
    df_filled = df.copy()
    
    # Fill numerical missing values with median
    if numerical_features:
        num_imputer = SimpleImputer(strategy='median')
        df_filled[numerical_features] = num_imputer.fit_transform(df[numerical_features])
    
    # Fill categorical missing values with most frequent value
    if categorical_features:
        for col in categorical_features:
            if df[col].isna().any():
                mode_value = df[col].mode()[0]
                df_filled[col] = df_filled[col].fillna(mode_value)
    
    return df_filled


def handle_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Handle outliers using z-score method.
    
    Args:
        df: Input dataframe
        threshold: Z-score threshold for outliers
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    numerical_features, _ = identify_features(df)
    
    df_no_outliers = df.copy()
    
    for col in numerical_features:
        if col == 'amount':  # Special handling for amount to preserve high value transactions
            continue
            
        # Calculate z-scores
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        
        # Cap outliers at threshold * std
        outlier_mask = z_scores > threshold
        if outlier_mask.any():
            cap_upper = df[col].mean() + threshold * df[col].std()
            cap_lower = df[col].mean() - threshold * df[col].std()
            
            df_no_outliers.loc[df[col] > cap_upper, col] = cap_upper
            df_no_outliers.loc[df[col] < cap_lower, col] = cap_lower
    
    return df_no_outliers


def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline for numerical and categorical features.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Numeric preprocessing: imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: imputation and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    return preprocessor


def balance_dataset(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple:
    """
    Balance dataset using SMOTE.
    
    Args:
        X: Feature dataframe
        y: Target series
        random_state: Random seed
        
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    return X_balanced, y_balanced


def preprocess_data(
    df: pd.DataFrame,
    scaler=None,
    encoder=None,
    training: bool = True
) -> pd.DataFrame:
    """
    Preprocess data for model training or prediction.
    
    Args:
        df: Input dataframe
        scaler: Fitted StandardScaler (for prediction mode)
        encoder: Fitted OneHotEncoder (for prediction mode)
        training: Whether preprocessing is for training
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Identify features
    numerical_features, categorical_features = identify_features(df)
    
    if training:
        # Create and fit the preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
        
        # Exclude the target from preprocessing
        if 'is_fraud' in df.columns:
            y = df['is_fraud']
            X = df.drop('is_fraud', axis=1)
        else:
            X = df
            y = None
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        
        # Get the feature names after preprocessing
        num_feature_names = numerical_features
        
        try:
            cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
        except:
            # Handle case where there are no categorical features
            cat_feature_names = []
        
        all_feature_names = list(num_feature_names) + list(cat_feature_names)
        
        # Convert to dataframe
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
        
        # Extract the fitted scaler and encoder for future use
        fitted_scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        fitted_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        
        if y is not None:
            return X_processed_df, y, fitted_scaler, fitted_encoder
        else:
            return X_processed_df, None, fitted_scaler, fitted_encoder
    else:
        # Prediction mode - use provided scaler and encoder
        if scaler is not None and encoder is not None:
            # Scale numerical features
            if len(numerical_features) > 0:
                df[numerical_features] = scaler.transform(df[numerical_features])
            
            # Encode categorical features
            if len(categorical_features) > 0:
                cat_encoded = encoder.transform(df[categorical_features])
                cat_feature_names = encoder.get_feature_names_out(categorical_features)
                cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)
                
                # Drop original categorical columns and add encoded ones
                df = df.drop(categorical_features, axis=1)
                df = pd.concat([df, cat_encoded_df], axis=1)
        
        return df 