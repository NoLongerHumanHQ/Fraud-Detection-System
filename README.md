# Fraud Detection System

A lightweight, production-ready fraud detection system with a Streamlit web interface. This system allows for real-time fraud prediction, batch processing, and model performance analysis.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and Isolation Forest
- **Interactive UI**: Intuitive Streamlit interface for fraud detection
- **Real-time Predictions**: Check individual transactions for fraud
- **Batch Processing**: Analyze multiple transactions at once
- **Visualization**: Transaction patterns and model performance metrics
- **Explainability**: Risk factors and fraud scores

## System Architecture

The system consists of several components:

1. **Data Processing Pipeline**: Preprocessing and feature engineering
2. **Machine Learning Models**: Multiple algorithms for fraud detection
3. **Web Interface**: Streamlit dashboard for interaction
4. **Visualization Layer**: Charts and graphs for analysis

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the models (optional, sample models are included):

```bash
python models/model_training.py
```

## Usage

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

This will launch the web interface on `http://localhost:8501`.

### Using the Interface

The interface includes several pages:

- **Home**: System overview and statistics
- **Single Prediction**: Check individual transactions
- **Batch Prediction**: Upload a CSV file for batch analysis
- **Model Performance**: Compare model metrics
- **Data Analysis**: Visualize transaction patterns
- **About**: System information and usage guide

### Single Transaction Analysis

1. Navigate to the "Single Prediction" page
2. Enter transaction details (amount, merchant, etc.)
3. Select a model for prediction
4. Click "Check Transaction"

The system will display a fraud score (0-100), risk category (Low, Medium, High), and key risk factors.

### Batch Analysis

1. Navigate to the "Batch Prediction" page
2. Upload a CSV file with transaction data
3. Select a model for prediction
4. Click "Analyze Batch"

The system will process all transactions and provide a summary of results, which can be downloaded as a CSV file.

## Data Format

For batch processing, your CSV file should include these columns:

- `transaction_id` (optional): Unique identifier
- `user_id`: User identifier
- `timestamp`: Transaction time (YYYY-MM-DD HH:MM:SS)
- `amount`: Transaction amount
- `merchant_category`: Category of the merchant
- `country`: Country where transaction occurred
- `city` (optional): City where transaction occurred
- `account_age`: Age of the account in days

Example:

```
transaction_id,user_id,timestamp,amount,merchant_category,country,city,account_age
T100001,U123456,2023-01-01 10:15:23,125.50,retail,USA,New York,365
T100002,U123456,2023-01-02 14:30:45,45.75,grocery,USA,New York,366
```

## Model Details

The system includes four fraud detection models:

1. **Logistic Regression**: A baseline model for binary classification
2. **Random Forest**: An ensemble method using decision trees
3. **XGBoost**: A gradient boosting algorithm known for its performance
4. **Isolation Forest**: An anomaly detection algorithm

## Customization

### Adding New Models

To add a new model:

1. Add model training code in `models/model_training.py`
2. Update the model loading function in `models/model_utils.py`
3. Add the model to the selection options in the Streamlit app

### Feature Engineering

Custom feature engineering can be added in `utils/feature_engineering.py`.

## Performance Optimization

- The system uses caching to avoid reloading models
- Batch predictions are optimized for memory efficiency
- The UI is responsive and works well on mobile devices

## Development

### Project Structure

```
fraud_detection_system/
├── app.py                 # Main Streamlit app
├── models/
│   ├── model_training.py  # Training pipeline
│   ├── model_utils.py     # Model utilities
│   └── saved_models/      # Serialized models
├── data/
│   ├── raw/              # Raw datasets
│   ├── processed/        # Cleaned data
│   └── sample_data.csv   # Sample for testing
├── utils/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── visualization.py
├── config/
│   └── config.py         # Configuration settings
├── requirements.txt
└── README.md
```

### Adding Features

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Inspired by real-world fraud detection systems
- Built with Streamlit, scikit-learn, and XGBoost
- Sample data based on common fraud patterns

## Contact

For questions or feedback, please open an issue on the GitHub repository. 