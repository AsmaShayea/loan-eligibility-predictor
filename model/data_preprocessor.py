import pandas as pd
import numpy as np

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Drop irrelevant columns
    if 'Loan_ID' in data.columns:
        data.drop(columns=['Loan_ID'], inplace=True)

    # Encode Loan_Status (Y → 1, N → 0)
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

    # Change column types
    data = data.astype({'Credit_History': object})

    # Handle missing values
    data.fillna({
        'Dependents': '0',                     
        'Self_Employed': 'No',                
        'Loan_Amount_Term': 360             
    }, inplace=True)

    # Handle missing Credit_History based on Loan_Status
    credit_loan = zip(data['Credit_History'], data['Loan_Status'])
    data['Credit_History'] = [
        0.0 if pd.isna(credit) and status == 0 else
        1.0 if pd.isna(credit) and status == 1 else
        credit for credit, status in credit_loan
    ]

    # Fill missing LoanAmount with median
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)

    # Drop rows with any remaining missing values
    print(f"Number of remaining missing values: {data.isna().sum().sum()}")
    data.dropna(axis=0, how='any', inplace=True)

    # Encode categorical features
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    data = pd.get_dummies(data=data, columns=categorical_cols, drop_first=True)

    return data


