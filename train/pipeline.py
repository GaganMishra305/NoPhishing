import os
import pickle
import numpy as np
import re
from urllib.parse import urlparse

import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier # Needed if XGBoost models are part of the pipeline

import warnings
warnings.filterwarnings('ignore')

class PhishingPredictionPipeline:
    def __init__(self, model, vectorizer, label_encoder, input_dim):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.input_dim = input_dim

    def _extract_url_features(self, url):
        features = []
        features.append(len(url))
        features.append(url.count('.'))
        features.append(url.count('/'))
        features.append(url.count('-'))
        features.append(url.count('_'))
        features.append(url.count('?'))
        features.append(url.count('='))
        features.append(url.count('&'))
        features.append(1 if 'https' in url else 0)
        features.append(1 if any(char.isdigit() for char in url) else 0)
        features.append(1 if re.search(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', url) else 0)
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            features.append(len(domain))
            features.append(domain.count('.'))
        except:
            features.append(0)
            features.append(0)
        return np.array([features], dtype=np.float32)

    def predict(self, url):
        if self.model is None or self.vectorizer is None or \
           self.label_encoder is None or self.input_dim is None:
            # This should ideally not happen if the pipeline was correctly saved/loaded
            print("Error: Pipeline components are not fully loaded. Cannot predict.")
            return None

        # Prepare URL features using the internal method
        url_features_np = self._extract_url_features(url)

        # Prepare TF-IDF features using the encapsulated vectorizer
        tfidf_features_np = self.vectorizer.transform([url]).toarray().astype(np.float32)

        # Concatenate all features to create the input for the model
        X_combined_np = np.hstack([url_features_np, tfidf_features_np])

        # Basic input shape validation (should match during training)
        if X_combined_np.shape[1] != self.input_dim:
            print(f"Error: Input feature dimension mismatch. Expected {self.input_dim}, got {X_combined_np.shape[1]}.")
            return None

        prediction_prob = None
        
        if isinstance(self.model, keras.Model):
            # Convert to TensorFlow tensor for Keras prediction
            X_inference = tf.constant(X_combined_np, dtype=tf.float32)
            
            # Reshape for CNN/RNN if the model name indicates it (matches architecture naming)
            if self.model.name.startswith("PhishingURL_CNN") or self.model.name.startswith("PhishingURL_RNN"):
                X_inference = tf.reshape(X_inference, (1, self.input_dim, 1))
            
            prediction_prob = self.model.predict(X_inference, verbose=0)[0][0]
        else: # Scikit-learn type model (e.g., XGBoost)
            prediction_prob = self.model.predict_proba(X_combined_np)[:, 1][0]
        
        prediction_binary = int(prediction_prob > 0.5)
        # Get the human-readable label from the encapsulated label encoder
        prediction_label = self.label_encoder.inverse_transform([prediction_binary])[0]

        # Determine confidence based on the predicted label
        # Assumes 'phishing' is the label encoded as 1 (positive class)
        positive_class_index = np.where(self.label_encoder.classes_ == 'phishing')[0][0] if 'phishing' in self.label_encoder.classes_ else 1

        confidence = float(prediction_prob if prediction_binary == positive_class_index else 1 - prediction_prob)

        return {
            'url': url,
            'prediction': prediction_label,
            'confidence': confidence
        }

