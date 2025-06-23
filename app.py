import os
import sys
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS # Required for Cross-Origin Resource Sharing

import warnings
warnings.filterwarnings('ignore')
from pipeline import PhishingPredictionPipeline 

app = Flask(__name__)
CORS(app) 

DEFAULT_MODEL_ARCHITECTURE = "ANN" # Default model to load on startup if not ensemble
SAVED_MODELS_BASE_DIR = "saved_models" # Relative to api/ directory

loaded_pipelines = {}
def load_pipeline_for_api(architecture_name, base_dir=SAVED_MODELS_BASE_DIR):
    """
    Loads a specific prediction pipeline from a .pkl file.
    """
    pipeline_path = os.path.join(base_dir, architecture_name.lower(), 
                                 f'{architecture_name.lower()}_pipeline.pkl')
    
    if not os.path.exists(pipeline_path):
        print(f"Error: Pipeline file not found at {pipeline_path} for {architecture_name}.")
        return None

    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"✅ Pipeline '{architecture_name}' loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"❌ Failed to load pipeline '{architecture_name}': {e}")
        return None

available_architectures = ["ANN", "CNN", "XGBoost"] # Match architectures you trained
for arch in available_architectures:
    pipeline = load_pipeline_for_api(arch, SAVED_MODELS_BASE_DIR)
    if pipeline:
        loaded_pipelines[arch.upper()] = pipeline

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    url = data.get('url')
    ensemble_prediction = data.get('ensemble', False) 
    if isinstance(ensemble_prediction, str): # Handle boolean passed as string from JS
        ensemble_prediction = ensemble_prediction.lower() == 'true'

    if not url:
        return jsonify({"error": "URL not provided"}), 400
    
    if ensemble_prediction:
        if not loaded_pipelines:
            return jsonify({"error": "No models loaded for ensemble prediction."}), 500

        all_predictions = []
        all_confidences = []
        
        for arch_name, pipeline in loaded_pipelines.items():
            try:
                result = pipeline.predict(url)
                if result:
                    if result['prediction'].lower() == 'phishing':
                        all_predictions.append(result['confidence'])
                    else:
                        all_predictions.append(1 - result['confidence']) # Convert legitimate confidence to phishing score
                else:
                    print(f"Warning: Prediction failed for {arch_name} on URL {url}. Skipping.")
            except Exception as e:
                print(f"Error predicting with {arch_name} model: {e}. Skipping this model for ensemble.")

        if not all_predictions:
            return jsonify({"error": "No successful predictions from any model for ensemble."}), 500

        # Simple averaging for ensemble score
        ensemble_score = sum(all_predictions) / len(all_predictions)
        
        ensemble_label = 'phishing' if ensemble_score > 0.5 else 'legitimate'
        ensemble_confidence = ensemble_score if ensemble_label == 'phishing' else (1 - ensemble_score)

        return jsonify({
            "url": url,
            "prediction": ensemble_label,
            "confidence": ensemble_confidence,
            "ensemble_members": list(loaded_pipelines.keys())
        }), 200

    else:
        # Single model prediction (existing behavior)
        architecture = data.get('architecture', DEFAULT_MODEL_ARCHITECTURE).upper() # Get architecture from request, default to ANN

        if architecture not in loaded_pipelines:
            return jsonify({"error": f"Model architecture '{architecture}' not loaded or invalid. Available: {list(loaded_pipelines.keys())}"}), 400

        pipeline = loaded_pipelines[architecture]

        try:
            prediction_result = pipeline.predict(url)
            if prediction_result:
                return jsonify(prediction_result), 200
            else:
                return jsonify({"error": "Prediction failed for the given URL"}), 500
        except Exception as e:
            print(f"Error during prediction for URL {url}: {e}")
            return jsonify({"error": "An internal error occurred during prediction."}), 500

@app.route('/', methods=['GET'])
def home():
    return "Phishing Detector API is running!"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
