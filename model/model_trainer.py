import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from model.data_preprocessor import preprocess_data

def train_model():
    
    loan_data = preprocess_data('loan_dataset.csv')

    if 'Loan_Status' not in loan_data.columns:
        raise ValueError("Loan_Status column is missing after preprocessing!")

    # Split data 
    X = loan_data.drop(columns=['Loan_Status'])
    y = loan_data['Loan_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    
    params = {
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 1, 5]
    }

    scorers = {
        'f1_score': make_scorer(f1_score),
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    # k-fold cross-validation
    skf = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scorers,
        n_jobs=-1,
        cv=skf,
        refit='accuracy_score'
    )

    # Train the model using grid search
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Ensure the bin directory exists before saving the model
    os.makedirs('bin', exist_ok=True)

    # Save the best model
    joblib.dump(best_model, 'bin/prediction_model.pkl')

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

