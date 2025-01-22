from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the pre-trained KNN model
model = joblib.load('knn_model.pkl')

@app.route('/')
def home():
    return "Flask backend is running!"

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()

        carbon = data.get('Carbon')
        organic_matter = data.get('Organic Matter')
        phosphorous = data.get('Phosphorous')
        calcium = data.get('Calcium')
        magnesium = data.get('Magnesium')
        potassium = data.get('Potassium')

        if None in [carbon, organic_matter, phosphorous, calcium, magnesium, potassium]:
            return jsonify({'error': 'Missing or invalid input values'}), 400

        # Prepare feature array
        features = np.array([[carbon, organic_matter, phosphorous, calcium, magnesium, potassium]])

        # Make a prediction
        predicted_crop = model.predict(features)[0]
        crop_name = "Soybean" if predicted_crop == 1 else "Paddy"

        return jsonify({'predicted_crop': crop_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host='0.0.0.0', port=port)
