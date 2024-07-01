import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge regressor model and standard scaler pickle
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading model/scaler files: {e}")
    raise

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Placeholder for additional features
            # Replace the placeholders with the actual form fields or default values
            Feature1 = 0.0  # Replace with actual feature value or request.form.get(...)
            Feature2 = 0.0  # Replace with actual feature value or request.form.get(...)
            Feature3 = 0.0  # Replace with actual feature value or request.form.get(...)

            # Combine all features into one array
            input_features = [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region, Feature1, Feature2, Feature3]

            # Scale the new data
            new_data_scaled = standard_scaler.transform([input_features])

            # Predict the result
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', result=result[0])

        except Exception as e:
            return render_template('home.html', result=f"Error: {e}")

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

