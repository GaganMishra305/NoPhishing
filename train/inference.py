import os
import pickle
import sys

SELECTED_MODEL_ARCHITECTURE = "ANN" # Adjust as needed: "ANN", "CNN", "RNN", "XGBoost"
SAVED_MODELS_BASE_DIR = "saved_models"

def main():
    pipeline_path = os.path.join(SAVED_MODELS_BASE_DIR, SELECTED_MODEL_ARCHITECTURE.lower(), 
                                 f'{SELECTED_MODEL_ARCHITECTURE.lower()}_pipeline.pkl')

    if not os.path.exists(pipeline_path):
        print(f"Error: Pipeline file not found at {pipeline_path}.")
        print("Please ensure the model was trained and saved using `main.py`.")
        sys.exit(1)

    # Load the complete pipeline
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)

    print(f"✅ Pipeline '{SELECTED_MODEL_ARCHITECTURE}' loaded successfully!")
    print("--- Phishing URL Predictor (Concise) ---")

    while True:
        user_url = input("Enter a URL to check (or 'quit'): ").strip()
        if user_url.lower() == 'quit':
            print("Exiting predictor. Goodbye!")
            break
        if not user_url:
            continue

        result = pipeline.predict(user_url)
        if result:
            print(f"URL: {result['url']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"Could not make a prediction for {user_url}.")
        print("-" * 50)

if __name__ == "__main__":
    main()

