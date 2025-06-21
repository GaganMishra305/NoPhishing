# main.py
import sys
import os
from train import PhishingURLDetector

def main():
    """
    Loads the trained Phishing URL Detector model and allows users to test URLs.
    """
    detector = PhishingURLDetector()
    model_files = {
        'model_path': 'phishing_model.h5',
        'vectorizer_path': 'vectorizer.pkl',
        'encoder_path': 'label_encoder.pkl'
    }

    print("Loading model...")
    if not detector.load_saved_model(**model_files):
        print("Error: Failed to load model components. Ensure they are in the same directory.")
        sys.exit(1)

    print("Model loaded. Enter URLs to check (type 'exit' to quit):")
    while True:
        url = input("URL: ").strip()
        if url.lower() == 'exit':
            print("Exiting.")
            break
        if not url:
            print("Please enter a URL.")
            continue

        try:
            result = detector.use(url)
            if result:
                print(f"  Prediction: {result['prediction']}, Confidence: {result['confidence']:.4f}")
            else:
                print(f"  Could not process URL: '{url}'.")
        except Exception as e:
            print(f"  Error processing URL: {e}")

if __name__ == "__main__":
    main()