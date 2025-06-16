"""
Feature engineering utilities for the fraud detection system.
"""
import pandas as pd
import numpy as np
from datetime import datetime


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamp column.
    
    Args:
        df: Input dataframe with timestamp column
        
    Returns:
        pd.DataFrame: Dataframe with additional time features
    """
    df = df.copy()
    
    # Check if timestamp column exists
    if 'timestamp' not in df.columns:
        return df
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            # If conversion fails, return original dataframe
            return df
    
    # Extract time features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create time bins
    conditions = [
        (df['hour_of_day'] >= 0) & (df['hour_of_day'] < 6),
        (df['hour_of_day'] >= 6) & (df['hour_of_day'] < 12),
        (df['hour_of_day'] >= 12) & (df['hour_of_day'] < 18),
        (df['hour_of_day'] >= 18) & (df['hour_of_day'] <= 23)
    ]
    
    choices = ['late_night', 'morning', 'afternoon', 'evening']
    df['time_of_day'] = pd.Series(np.select(conditions, choices, 'unknown'), index=df.index)
    
    return df


def calculate_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transaction velocity features.
    
    These features help detect unusual patterns in transaction frequency.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional velocity features
    """
    df = df.copy()
    
    # Check if timestamp and user_id columns exist
    if 'timestamp' not in df.columns or 'user_id' not in df.columns:
        return df
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            return df
    
    # Sort by user_id and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Calculate time difference between consecutive transactions for each user
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    
    # Time difference in hours
    df['time_since_prev_tx'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600
    
    # Fill NaN (for first transaction of each user) with a high value
    df['time_since_prev_tx'] = df['time_since_prev_tx'].fillna(24 * 7)  # 1 week
    
    # Create velocity flags
    df['high_velocity'] = (df['time_since_prev_tx'] < 1).astype(int)  # Less than 1 hour between transactions
    
    # Clean up intermediate columns
    df = df.drop('prev_timestamp', axis=1)
    
    return df


def calculate_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on transaction amount.
    
    Args:
        df: Input dataframe with amount column
        
    Returns:
        pd.DataFrame: Dataframe with additional amount features
    """
    df = df.copy()
    
    # Check if amount column exists
    if 'amount' not in df.columns:
        return df
    
    # Check if user_id column exists for user-level features
    has_user_id = 'user_id' in df.columns
    
    # Create flags for different amount thresholds
    df['is_high_amount'] = (df['amount'] > 1000).astype(int)
    df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
    
    # Create additional user-level amount features if possible
    if has_user_id:
        # Calculate user average transaction amount
        user_avg = df.groupby('user_id')['amount'].transform('mean')
        user_std = df.groupby('user_id')['amount'].transform('std')
        
        # Handle users with only one transaction (std = NaN)
        user_std = user_std.fillna(0)
        
        # Calculate z-score relative to user's history
        df['amount_user_zscore'] = (df['amount'] - user_avg) / (user_std + 1)  # Add 1 to avoid division by zero
        
        # Flag for unusual amounts (> 2 std from user's mean)
        df['unusual_amount'] = (abs(df['amount_user_zscore']) > 2).astype(int)
    
    return df


def calculate_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on location data.
    
    Args:
        df: Input dataframe with location information
        
    Returns:
        pd.DataFrame: Dataframe with additional location features
    """
    df = df.copy()
    
    # Check for location columns
    has_location = 'country' in df.columns
    has_user_id = 'user_id' in df.columns
    
    if not has_location or not has_user_id:
        return df
    
    # Flag for transactions in unusual countries for each user
    if has_location and has_user_id:
        # Find most frequent country for each user
        user_common_country = df.groupby('user_id')['country'].transform(
            lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
        )
        
        # Flag for transactions in a different country than the user's most common one
        df['unusual_country'] = (df['country'] != user_common_country).astype(int)
    
    return df


def calculate_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on account information.
    
    Args:
        df: Input dataframe with account_age column
        
    Returns:
        pd.DataFrame: Dataframe with additional account features
    """
    df = df.copy()
    
    # Check if account_age column exists
    if 'account_age' not in df.columns:
        return df
    
    # Create flags for different account age thresholds
    df['new_account'] = (df['account_age'] < 30).astype(int)  # Less than 30 days
    df['established_account'] = (df['account_age'] > 365).astype(int)  # More than 1 year
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the input dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with all engineered features
    """
    df = df.copy()
    
    # Apply feature engineering functions in sequence
    df = extract_time_features(df)
    df = calculate_transaction_velocity(df)
    df = calculate_amount_features(df)
    df = calculate_location_features(df)
    df = calculate_account_features(df)
    
    return df 