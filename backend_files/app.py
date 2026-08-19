from flask import Flask, request, jsonify
import pandas as pd
import joblib  # or your preferred model loading library

superkart_api = Flask(__name__)

# Load model globally once on startup
# model = joblib.load('model.pkl') 

# ------------------------------------------------------------------
# Single Inference Endpoint
# ------------------------------------------------------------------
@superkart_api.post('/v1/predict')
def predict_single():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request payload'}), 400

    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        
        # Convert numpy types to native Python types for JSON serialization if needed
        return jsonify({'prediction': float(prediction)}), 200
    except Exception as e:
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500


# ------------------------------------------------------------------
# Batch Inference Endpoint
# ------------------------------------------------------------------
@superkart_api.post('/v1/predictbatch')
def predict_batch():
    data = request.get_json()

    if not data or 'instances' not in data:
        return jsonify({'error': 'Missing "instances" key in request body'}), 400
    
    instances = data['instances']
    
    if not isinstance(instances, list) or len(instances) == 0:
        return jsonify({'error': '"instances" must be a non-empty list'}), 400

    try:
        df = pd.DataFrame(instances)
        predictions = model.predict(df).tolist()
        
        return jsonify({'predictions': predictions}), 200
    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500


# ------------------------------------------------------------------
# Server Runner
# ------------------------------------------------------------------
if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0', port=5000, debug=True)
