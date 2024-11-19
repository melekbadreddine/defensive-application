from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        comp1 = float(request.form['component1'])
        comp2 = float(request.form['component2'])

        # Prepare the feature vector
        features = np.array([comp1, comp2]).reshape(1, -1)

        # Make prediction
        prediction = rf_model.predict(features)

        # Interpret the result
        result = "Attack" if prediction[0] == 1 else "Normal"

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
