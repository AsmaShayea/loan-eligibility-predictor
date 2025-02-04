# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import os
# from flask_cors import CORS

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Get the current directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Load trained model
# model_path = os.path.join(current_dir, "bin/prediction_model.pkl")
# model = joblib.load(model_path)

# # Define JSON schema for column names
# schema_cols = [
#     "ApplicantIncome",
#     "CoapplicantIncome",
#     "LoanAmount",
#     "Loan_Amount_Term",
#     "Gender_Male",
#     "Married_Yes",
#     "Dependents_1",
#     "Dependents_2",
#     "Dependents_3+",
#     "Education_Not Graduate",
#     "Self_Employed_Yes",
#     "Credit_History_1.0",
#     "Property_Area_Semiurban",
#     "Property_Area_Urban"
# ]

# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Loan Prediction API is running!"})

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get JSON data from request
#         data = request.json

#         # Create a dictionary with default values
#         input_data = {col: 0 for col in schema_cols}

#         # Parse user input and update the input_data dictionary
#         input_data["ApplicantIncome"] = float(data["applicant_income"])
#         input_data["CoapplicantIncome"] = float(data["coapplicant_income"])
#         input_data["LoanAmount"] = float(data["loan_amount"])
#         input_data["Loan_Amount_Term"] = float(data["loan_term"])
#         input_data["Gender_Male"] = int(data["gender"])
#         input_data["Married_Yes"] = int(data["marital_status"])
#         input_data["Education_Not Graduate"] = int(data["education"])
#         input_data["Self_Employed_Yes"] = int(data["self_employed"])
#         input_data["Credit_History_1.0"] = int(data["credit_history"])

#         # Handle categorical fields (Dependents & Property_Area)
#         dependents_col = f"Dependents_{data['dependents']}"
#         property_area_col = f"Property_Area_{data['property_area']}"

#         if dependents_col in input_data:
#             input_data[dependents_col] = 1
#         if property_area_col in input_data:
#             input_data[property_area_col] = 1

#         # Convert to DataFrame
#         input_df = pd.DataFrame([input_data])

#         # Make prediction
#         prediction = model.predict(input_df)[0]
#         result = "approved" if prediction == 1 else "rejected"

#         return jsonify({"loan_status": result})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == "__main__":
#     app.run(debug=True)


import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model_trainer import train_model
from model.loan_predictor import predict_loan_status

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js integration

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Loan Prediction API is running!"})

@app.route("/train", methods=["GET"])
def train():
    train_model()
    return jsonify({'message': 'Model training complete'})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Call prediction function
        result = predict_loan_status(data)

        return jsonify({"loan_status": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
