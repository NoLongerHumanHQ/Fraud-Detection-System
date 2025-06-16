import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from utils.data_preprocessing import preprocess_data
from utils.feature_engineering import engineer_features
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_fraud_distribution,
    plot_fraud_amount_distribution
)
from config.config import MODEL_PATH, SAMPLE_DATA_PATH

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'logistic_regression': 'logistic_regression_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'isolation_forest': 'isolation_forest_model.pkl',
        'scaler': 'scaler.pkl',
        'encoder': 'encoder.pkl'
    }
    
    try:
        for model_name, file_name in model_files.items():
            with open(f"{MODEL_PATH}/{file_name}", 'rb') as f:
                models[model_name] = pickle.load(f)
    except FileNotFoundError:
        st.error("Model files not found. Please train the models first.")
    
    return models

# Load sample data
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv(SAMPLE_DATA_PATH)
    except FileNotFoundError:
        st.error("Sample data not found.")
        return pd.DataFrame()

# Calculate fraud score (0-100)
def calculate_fraud_score(probability):
    return int(probability * 100)

# Determine risk category
def get_risk_category(score):
    if score < 30:
        return "Low", "green"
    elif score < 70:
        return "Medium", "orange"
    else:
        return "High", "red"

# Predict for a single transaction
def predict_transaction(transaction_data, models, model_choice):
    # Preprocess the data
    df = pd.DataFrame([transaction_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Preprocess
    X = preprocess_data(df, models['scaler'], models['encoder'], training=False)
    
    # Make prediction
    if model_choice == 'isolation_forest':
        # For isolation forest, negative scores indicate anomalies
        score = models[model_choice].decision_function([X.iloc[0].values])
        # Convert to probability-like score (0 to 1)
        probability = 1 - (score[0] + 0.5)  # Normalize from [-0.5, 0.5] to [1, 0]
        probability = max(0, min(probability, 1))  # Clamp between 0 and 1
    else:
        probability = models[model_choice].predict_proba([X.iloc[0].values])[0][1]
    
    fraud_score = calculate_fraud_score(probability)
    risk_category, color = get_risk_category(fraud_score)
    
    return probability, fraud_score, risk_category, color

# Predict for batch transactions
def predict_batch(data, models, model_choice):
    # Engineer features
    data = engineer_features(data)
    
    # Preprocess
    X = preprocess_data(data, models['scaler'], models['encoder'], training=False)
    
    # Make predictions
    if model_choice == 'isolation_forest':
        # For isolation forest
        scores = models[model_choice].decision_function(X.values)
        probabilities = 1 - (scores + 0.5)  # Normalize
        probabilities = np.clip(probabilities, 0, 1)  # Clamp between 0 and 1
    else:
        probabilities = models[model_choice].predict_proba(X.values)[:, 1]
    
    # Add results to dataframe
    results = data.copy()
    results['fraud_probability'] = probabilities
    results['fraud_score'] = results['fraud_probability'].apply(calculate_fraud_score)
    results['risk_category'] = results['fraud_score'].apply(lambda x: get_risk_category(x)[0])
    
    return results

def main():
    # Sidebar
    st.sidebar.title("Fraud Detection System")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Single Prediction", "Batch Prediction", "Model Performance", "Data Analysis", "About"]
    )
    
    # Load models and data
    models = load_models()
    sample_data = load_sample_data()
    
    # Pages
    if page == "Home":
        st.title("Fraud Detection System")
        st.subheader("Welcome to the Fraud Detection Dashboard")
        
        # System overview
        st.write("""
        This system uses machine learning to detect potentially fraudulent transactions.
        Use the sidebar to navigate between different features.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("### Key Statistics")
            if not sample_data.empty:
                st.metric("Total Transactions", f"{len(sample_data):,}")
                fraud_count = sample_data['is_fraud'].sum()
                st.metric("Fraud Transactions", f"{fraud_count:,}")
                st.metric("Fraud Rate", f"{fraud_count / len(sample_data):.2%}")
        
        with col2:
            st.info("### Quick Actions")
            st.button("⚡ Check a Transaction", on_click=lambda: st.session_state.update({"page": "Single Prediction"}))
            st.button("📊 View Model Performance", on_click=lambda: st.session_state.update({"page": "Model Performance"}))
        
        # Sample visualization
        st.subheader("Sample Data Preview")
        if not sample_data.empty:
            st.dataframe(sample_data.head(10))
            
            st.subheader("Fraud Distribution")
            fig = plot_fraud_distribution(sample_data)
            st.plotly_chart(fig)
            
    elif page == "Single Prediction":
        st.title("Single Transaction Fraud Check")
        st.write("Enter transaction details to check for potential fraud")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=100.0)
            
            # Current timestamp
            now = datetime.datetime.now()
            date = st.date_input("Transaction Date", now.date())
            time = st.time_input("Transaction Time", now.time())
            timestamp = datetime.datetime.combine(date, time)
            
            merchant_category = st.selectbox(
                "Merchant Category",
                ["retail", "grocery", "entertainment", "travel", "restaurant", "technology", "healthcare"]
            )
            
        with col2:
            user_id = st.text_input("User ID", value="U123456")
            
            # Simplified location selection
            country = st.selectbox("Country", ["USA", "Canada", "UK", "Australia", "Germany", "France", "Other"])
            city = st.text_input("City", value="New York")
            
            # Account age in days
            account_age = st.slider("Account Age (days)", 1, 3650, 365)
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["logistic_regression", "random_forest", "xgboost", "isolation_forest"]
        )
        
        # Transaction data
        transaction_data = {
            'amount': amount,
            'timestamp': timestamp,
            'merchant_category': merchant_category,
            'user_id': user_id,
            'country': country,
            'city': city,
            'account_age': account_age
        }
        
        if st.button("Check Transaction"):
            if models:
                with st.spinner("Analyzing transaction..."):
                    probability, fraud_score, risk_category, color = predict_transaction(
                        transaction_data, models, model_choice
                    )
                
                # Display results
                st.subheader("Fraud Analysis Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fraud Score", f"{fraud_score}/100")
                
                with col2:
                    st.markdown(f"<h3 style='color:{color}'>Risk: {risk_category}</h3>", unsafe_allow_html=True)
                
                with col3:
                    st.metric("Fraud Probability", f"{probability:.4f}")
                
                # Explanation
                st.subheader("Key Risk Factors")
                
                # Simplified explanation factors for demo
                factors = [
                    {"factor": "Transaction Amount", "impact": "High" if amount > 1000 else "Low"},
                    {"factor": "Transaction Time", "impact": "Medium" if timestamp.hour < 6 or timestamp.hour > 22 else "Low"},
                    {"factor": "Account Age", "impact": "High" if account_age < 30 else "Low"},
                    {"factor": "Location", "impact": "Medium" if country == "Other" else "Low"}
                ]
                
                for factor in factors:
                    st.write(f"**{factor['factor']}**: {factor['impact']} impact")
    
    elif page == "Batch Prediction":
        st.title("Batch Transaction Analysis")
        st.write("Upload a CSV file with multiple transactions to analyze")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["logistic_regression", "random_forest", "xgboost", "isolation_forest"]
        )
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(batch_data.head())
                
                if st.button("Analyze Batch"):
                    if models:
                        with st.spinner("Analyzing transactions..."):
                            results = predict_batch(batch_data, models, model_choice)
                        
                        st.success(f"Analysis complete for {len(results)} transactions")
                        
                        # Display results
                        st.subheader("Results Overview")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            high_risk = (results['risk_category'] == 'High').sum()
                            st.metric("High Risk Transactions", high_risk)
                        
                        with col2:
                            medium_risk = (results['risk_category'] == 'Medium').sum()
                            st.metric("Medium Risk Transactions", medium_risk)
                        
                        with col3:
                            low_risk = (results['risk_category'] == 'Low').sum()
                            st.metric("Low Risk Transactions", low_risk)
                        
                        # Show results table
                        st.subheader("Detailed Results")
                        st.dataframe(results)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="fraud_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        st.subheader("Risk Distribution")
                        
                        fig = px.pie(
                            results,
                            names='risk_category',
                            title="Risk Categories Distribution"
                        )
                        st.plotly_chart(fig)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        # Demo with sample data
        st.divider()
        st.subheader("Or try with sample data")
        
        if st.button("Use Sample Data"):
            if not sample_data.empty and models:
                with st.spinner("Analyzing sample transactions..."):
                    results = predict_batch(sample_data, models, model_choice)
                
                st.success(f"Analysis complete for {len(results)} sample transactions")
                
                # Display results
                st.subheader("Results Overview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_risk = (results['risk_category'] == 'High').sum()
                    st.metric("High Risk Transactions", high_risk)
                
                with col2:
                    medium_risk = (results['risk_category'] == 'Medium').sum()
                    st.metric("Medium Risk Transactions", medium_risk)
                
                with col3:
                    low_risk = (results['risk_category'] == 'Low').sum()
                    st.metric("Low Risk Transactions", low_risk)
                
                # Show results table
                st.subheader("Detailed Results")
                st.dataframe(results)
                
                # Visualization
                st.subheader("Risk Distribution")
                
                fig = px.pie(
                    results,
                    names='risk_category',
                    title="Risk Categories Distribution"
                )
                st.plotly_chart(fig)
    
    elif page == "Model Performance":
        st.title("Model Performance")
        
        model_choice = st.selectbox(
            "Select Model to Evaluate",
            ["logistic_regression", "random_forest", "xgboost", "isolation_forest"]
        )
        
        if models:
            # Display model metrics (in real implementation, these would be loaded from stored evaluation results)
            st.subheader("Performance Metrics")
            
            # Example metrics (would be replaced with actual model evaluation)
            metrics = {
                'logistic_regression': {
                    'accuracy': 0.92, 'precision': 0.86, 'recall': 0.79, 'f1': 0.82, 'auc': 0.91
                },
                'random_forest': {
                    'accuracy': 0.96, 'precision': 0.91, 'recall': 0.85, 'f1': 0.88, 'auc': 0.95
                },
                'xgboost': {
                    'accuracy': 0.97, 'precision': 0.92, 'recall': 0.86, 'f1': 0.89, 'auc': 0.96
                },
                'isolation_forest': {
                    'accuracy': 0.91, 'precision': 0.84, 'recall': 0.76, 'f1': 0.80, 'auc': 0.88
                }
            }
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            model_metrics = metrics[model_choice]
            
            with col1:
                st.metric("Accuracy", f"{model_metrics['accuracy']:.2f}")
            
            with col2:
                st.metric("Precision", f"{model_metrics['precision']:.2f}")
            
            with col3:
                st.metric("Recall", f"{model_metrics['recall']:.2f}")
            
            with col4:
                st.metric("F1 Score", f"{model_metrics['f1']:.2f}")
            
            with col5:
                st.metric("AUC-ROC", f"{model_metrics['auc']:.2f}")
            
            # Visual metrics
            st.subheader("Visual Evaluation")
            
            tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])
            
            with tab1:
                # Example confusion matrix (would be generated from actual evaluation)
                fig = plot_confusion_matrix(model_choice)
                st.pyplot(fig)
            
            with tab2:
                # Example ROC curve
                fig = plot_roc_curve(model_choice)
                st.pyplot(fig)
            
            with tab3:
                # Example feature importance
                fig = plot_feature_importance(model_choice)
                st.pyplot(fig)
            
            # Model comparison
            st.subheader("Model Comparison")
            
            comparison_df = pd.DataFrame(metrics).T.reset_index()
            comparison_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            
            fig = px.bar(
                comparison_df,
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                title="Performance Comparison Across Models",
                barmode='group'
            )
            st.plotly_chart(fig)
    
    elif page == "Data Analysis":
        st.title("Transaction Data Analysis")
        
        if not sample_data.empty:
            # Fraud distribution
            st.subheader("Fraud Distribution")
            fig = plot_fraud_distribution(sample_data)
            st.plotly_chart(fig)
            
            # Transaction amount distribution by fraud/non-fraud
            st.subheader("Transaction Amount Distribution")
            fig = plot_fraud_amount_distribution(sample_data)
            st.plotly_chart(fig)
            
            # Transaction time patterns
            st.subheader("Transaction Time Patterns")
            
            # Add hour of day column if not present
            if 'hour_of_day' not in sample_data.columns and 'timestamp' in sample_data.columns:
                sample_data['hour_of_day'] = pd.to_datetime(sample_data['timestamp']).dt.hour
            
            if 'hour_of_day' in sample_data.columns:
                hourly_fraud = sample_data.groupby('hour_of_day')['is_fraud'].mean().reset_index()
                
                fig = px.line(
                    hourly_fraud,
                    x='hour_of_day',
                    y='is_fraud',
                    title="Fraud Rate by Hour of Day",
                    labels={'hour_of_day': 'Hour of Day', 'is_fraud': 'Fraud Rate'}
                )
                st.plotly_chart(fig)
            
            # Category-based analysis
            if 'merchant_category' in sample_data.columns:
                st.subheader("Merchant Category Analysis")
                
                category_fraud = sample_data.groupby('merchant_category')['is_fraud'].agg(['mean', 'count']).reset_index()
                category_fraud.columns = ['Merchant Category', 'Fraud Rate', 'Transaction Count']
                
                fig = px.scatter(
                    category_fraud,
                    x='Transaction Count',
                    y='Fraud Rate',
                    size='Transaction Count',
                    color='Fraud Rate',
                    hover_name='Merchant Category',
                    title="Fraud Rate by Merchant Category"
                )
                st.plotly_chart(fig)
    
    elif page == "About":
        st.title("About the Fraud Detection System")
        
        st.markdown("""
        ### System Overview
        
        This fraud detection system uses machine learning algorithms to identify potentially fraudulent transactions.
        It provides real-time predictions for individual transactions and batch analysis for multiple transactions.
        
        ### Models Used
        
        - **Logistic Regression**: A baseline model for binary classification
        - **Random Forest**: An ensemble method using decision trees
        - **XGBoost**: A gradient boosting algorithm known for its performance
        - **Isolation Forest**: An anomaly detection algorithm
        
        ### Key Features
        
        - Real-time fraud scoring (0-100 scale)
        - Risk categorization (Low, Medium, High)
        - Batch transaction analysis
        - Model performance comparison
        - Transaction pattern visualization
        
        ### How to Use
        
        1. Use the **Single Prediction** page to check individual transactions
        2. Use the **Batch Prediction** page to analyze multiple transactions at once
        3. View model performance metrics on the **Model Performance** page
        4. Explore transaction patterns on the **Data Analysis** page
        
        ### Data Privacy
        
        This system only processes data provided by the user and does not store any sensitive information.
        All analyses are performed locally on your device.
        """)

if __name__ == "__main__":
    main() 