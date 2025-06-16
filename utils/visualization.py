"""
Visualization utilities for the fraud detection system.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(model_name: str) -> plt.Figure:
    """
    Plot confusion matrix for a model.
    
    Args:
        model_name: Name of the model to plot
        
    Returns:
        plt.Figure: Matplotlib figure with the confusion matrix plot
    """
    # Sample confusion matrix for visualization purposes
    # In a real implementation, this would use actual evaluation results
    if model_name == 'logistic_regression':
        cm = np.array([[955, 45], [21, 79]])
    elif model_name == 'random_forest':
        cm = np.array([[970, 30], [15, 85]])
    elif model_name == 'xgboost':
        cm = np.array([[975, 25], [14, 86]])
    elif model_name == 'isolation_forest':
        cm = np.array([[950, 50], [24, 76]])
    else:
        # Default confusion matrix
        cm = np.array([[960, 40], [20, 80]])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Not Fraud', 'Fraud'],
        yticklabels=['Not Fraud', 'Fraud']
    )
    
    # Add labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
    
    # Improve layout
    plt.tight_layout()
    
    return fig


def plot_roc_curve(model_name: str) -> plt.Figure:
    """
    Plot ROC curve for a model.
    
    Args:
        model_name: Name of the model to plot
        
    Returns:
        plt.Figure: Matplotlib figure with the ROC curve plot
    """
    # Sample ROC curve data for visualization purposes
    # In a real implementation, this would use actual evaluation results
    if model_name == 'logistic_regression':
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 1/3)
    elif model_name == 'random_forest':
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 1/5)
    elif model_name == 'xgboost':
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 1/6)
    elif model_name == 'isolation_forest':
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 1/2.5)
    else:
        # Default ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 1/4)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(
        fpr, 
        tpr, 
        color='darkorange',
        lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.2f})'
    )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name.replace("_", " ").title()}')
    plt.legend(loc="lower right")
    
    # Improve layout
    plt.tight_layout()
    
    return fig


def plot_feature_importance(model_name: str) -> plt.Figure:
    """
    Plot feature importance for a model.
    
    Args:
        model_name: Name of the model to plot
        
    Returns:
        plt.Figure: Matplotlib figure with the feature importance plot
    """
    # Sample feature importance data for visualization purposes
    # In a real implementation, this would use actual model feature importances
    features = [
        'amount', 
        'hour_of_day', 
        'is_high_amount', 
        'unusual_amount', 
        'time_since_prev_tx',
        'high_velocity',
        'is_weekend',
        'unusual_country',
        'new_account'
    ]
    
    if model_name == 'logistic_regression':
        importances = [0.25, 0.15, 0.20, 0.18, 0.08, 0.05, 0.03, 0.04, 0.02]
    elif model_name == 'random_forest':
        importances = [0.22, 0.18, 0.17, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02]
    elif model_name == 'xgboost':
        importances = [0.20, 0.18, 0.15, 0.15, 0.12, 0.10, 0.05, 0.03, 0.02]
    elif model_name == 'isolation_forest':
        importances = [0.30, 0.15, 0.20, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02]
    else:
        # Default feature importance
        importances = [0.25, 0.15, 0.15, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center', color='skyblue')
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
    ax.invert_yaxis()  # Feature with highest importance at the top
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {model_name.replace("_", " ").title()}')
    
    # Improve layout
    plt.tight_layout()
    
    return fig


def plot_fraud_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plot fraud distribution using Plotly.
    
    Args:
        df: Dataframe containing fraud data
        
    Returns:
        go.Figure: Plotly figure with the fraud distribution
    """
    # Check if is_fraud column exists
    if 'is_fraud' not in df.columns:
        # Create a placeholder figure
        fig = px.bar(
            x=['No Data'], 
            y=[0], 
            title="No Fraud Data Available",
            labels={'x': '', 'y': 'Count'}
        )
        return fig
    
    # Create fraud counts
    fraud_counts = df['is_fraud'].value_counts().reset_index()
    fraud_counts.columns = ['Fraud', 'Count']
    fraud_counts['Fraud'] = fraud_counts['Fraud'].map({0: 'Legitimate', 1: 'Fraud'})
    
    # Create plot
    fig = px.bar(
        fraud_counts,
        x='Fraud',
        y='Count',
        color='Fraud',
        color_discrete_map={'Legitimate': 'green', 'Fraud': 'red'},
        title='Distribution of Fraudulent vs Legitimate Transactions',
        text='Count'
    )
    
    # Update layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Transaction Type",
        yaxis_title="Number of Transactions",
        legend_title="Transaction Type"
    )
    
    return fig


def plot_fraud_amount_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plot transaction amount distribution by fraud status using Plotly.
    
    Args:
        df: Dataframe containing fraud and amount data
        
    Returns:
        go.Figure: Plotly figure with the amount distribution
    """
    # Check if required columns exist
    if 'is_fraud' not in df.columns or 'amount' not in df.columns:
        # Create a placeholder figure
        fig = px.histogram(
            x=['No Data'], 
            y=[0], 
            title="No Amount Data Available",
            labels={'x': '', 'y': 'Count'}
        )
        return fig
    
    # Create fraud label
    df_plot = df.copy()
    df_plot['fraud_status'] = df_plot['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
    
    # Create histogram
    fig = px.histogram(
        df_plot,
        x='amount',
        color='fraud_status',
        nbins=50,
        opacity=0.7,
        barmode='overlay',
        color_discrete_map={'Legitimate': 'green', 'Fraud': 'red'},
        title='Transaction Amount Distribution by Fraud Status',
        labels={'amount': 'Transaction Amount', 'count': 'Number of Transactions'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Transaction Amount",
        yaxis_title="Number of Transactions",
        legend_title="Transaction Type"
    )
    
    return fig


def plot_fraud_time_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Plot fraud rate heatmap by hour and day of week using Plotly.
    
    Args:
        df: Dataframe containing fraud and time data
        
    Returns:
        go.Figure: Plotly figure with the time heatmap
    """
    # Check if required columns exist
    required_cols = ['is_fraud', 'hour_of_day', 'day_of_week']
    if not all(col in df.columns for col in required_cols):
        # Add time features if timestamp is available
        if 'timestamp' in df.columns:
            df = df.copy()
            df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        else:
            # Create a placeholder figure
            fig = px.imshow(
                np.zeros((1, 1)), 
                title="No Time Data Available",
                labels={'x': '', 'y': ''}
            )
            return fig
    
    # Ensure all hours (0-23) and days (0-6) are represented
    hours = list(range(24))
    days = list(range(7))
    
    # Create a grid for all hours and days combinations
    grid = pd.DataFrame([(h, d) for h in hours for d in days], columns=['hour_of_day', 'day_of_week'])
    
    # Calculate fraud rate by hour and day
    fraud_by_time = df.groupby(['hour_of_day', 'day_of_week'])['is_fraud'].agg(['mean', 'count']).reset_index()
    
    # Merge with the grid to ensure all combinations exist
    fraud_grid = grid.merge(fraud_by_time, on=['hour_of_day', 'day_of_week'], how='left').fillna(0)
    
    # Reshape to create a matrix for heatmap
    fraud_matrix = fraud_grid.pivot_table(
        index='day_of_week', 
        columns='hour_of_day', 
        values='mean'
    ).values
    
    # Create day of week labels
    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create heatmap
    fig = px.imshow(
        fraud_matrix,
        x=hours,
        y=day_labels,
        color_continuous_scale='Reds',
        labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Fraud Rate'},
        title='Fraud Rate by Hour and Day'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week"
    )
    
    return fig 