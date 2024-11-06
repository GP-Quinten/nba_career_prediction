from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from config import MODELS_DIR

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib')  # Replace with actual model name if different
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features_list = model_data['features_list']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_features = [data[feature] for feature in features_list]
        
        # Scale and predict
        input_scaled = scaler.transform([input_features])
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0, 1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': prediction_proba
        })
    except KeyError as e:
        return jsonify({'error': f'Missing feature in input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
