# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKartSalesAPI")

# Load the trained model
model = joblib.load("xgb_tuned_model.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint to predict sales for a single item
@superkart_api.post('/v1/predict')
def predict_sales():
    data = request.get_json()

    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    input_data = pd.DataFrame([sample])
    prediction = model.predict(input_data).tolist()[0]

    return jsonify({'Sales': prediction})

if __name__ == '__main__':
    # BIND TO 0.0.0.0 TO ALLOW DOCKER & CODESPACES REVERSE PROXY ACCESS
    superkart_api.run(host='0.0.0.0', port=5000, debug=True)
