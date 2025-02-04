# Loan Eligibility Predictor App

This is a Flask-based web application that uses XGBoost to predict loan eligibility with approximately 84% accuracy. 

## Features:
- Predicts loan eligibility based on user-provided inputs.
- Preprocessing includes handling missing values, encodes categorical variables.
- Hyperparameter tuning with GridSearchCV and 10-fold cross-validation for model optimization.
- Utilizes XGBoost for robust and accurate predictions.

## Directory Structure:
- **`app.py`**: Main Flask application.
- **`model/model_trainer.py`**: Trains the XGBoost model.
- **`model/loan_predictor.py`**: Handles prediction logic.
- **`model/data_preprocessor.py`**: Prepares data for training and prediction.
- **`prediction_model.pkl`**: Serialized pre-trained model.
