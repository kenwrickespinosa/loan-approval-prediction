import sys
import json
import joblib
import numpy as np

# Load model
model = joblib.load("predmodels/XGBoost.joblib")

# Read input passed from Laravel
raw = sys.stdin.read()
data = json.loads(raw)

# Convert values to model input order
features = np.array([[
    data["Gender"],
    data["Married"],
    data["Dependents"],
    data["Education"],
    data["Self_Employed"],
    data["ApplicantIncome"],
    data["CoapplicantIncome"],
    data["LoanAmount"],
    data["Loan_Amount_Term"],
    data["Credit_History"],
    data["Property_Area"]
]])

# Predict
pred = model.predict(features)[0]

# Return back to Laravel
print(json.dumps({"Prediction": int(pred)}))