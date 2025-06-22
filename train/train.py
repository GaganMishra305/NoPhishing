### this file was not used for the final training ###

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import re
import os
import signal
import sys
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress specific future warnings from TensorFlow
warnings.filterwarnings('ignore', category=FutureWarning)
# Suppress general warnings that might clutter output
warnings.filterwarnings('ignore')

class PhishingURLDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.max_features = 5000
        self.training_history = None
        self.validation_data = None # Will store X_val, y_val as tf.Tensors
        self.input_dim = None # To store the input dimension derived from data
        self.interrupt_handler_set = False

        self.setup_interrupt_handler()

    def extract_url_features(self, url):
        """Extracts numerical features from a given URL."""
        features = []

        # Length-based features
        features.append(len(url))
        features.append(url.count('.'))
        features.append(url.count('/'))
        features.append(url.count('-'))
        features.append(url.count('_'))
        features.append(url.count('?'))
        features.append(url.count('='))
        features.append(url.count('&'))

        # Presence of common indicators
        features.append(1 if 'https' in url else 0)
        features.append(1 if any(char.isdigit() for char in url) else 0)
        # Presence of IP address in hostname
        features.append(1 if re.search(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', url) else 0)

        # Parsed URL features
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            features.append(len(domain))
            features.append(domain.count('.'))
        except:
            features.append(0) # Default if parsing fails
            features.append(0)
        
        return features

    def setup_interrupt_handler(self):
        """Sets up a signal handler to gracefully save the model on Ctrl+C."""
        if not self.interrupt_handler_set:
            def signal_handler(sig, frame):
                print('\n⚠️  Keyboard interrupt detected!')
                if self.model is not None:
                    print('💾 Saving model before exit...')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    try:
                        self.save_model(
                            model_path=f'interrupted_model_{timestamp}.h5',
                            vectorizer_path=f'interrupted_vectorizer_{timestamp}.pkl',
                            encoder_path=f'interrupted_encoder_{timestamp}.pkl'
                        )
                        print('✅ Model saved successfully!')
                    except Exception as e:
                        print(f'❌ Error saving model: {e}')
                else:
                    print('❌ No model to save.')
                print('👋 Exiting...')
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            self.interrupt_handler_set = True

    def plot(self, save_path='plot.png'):
        """Plots training history (loss, accuracy, precision, recall)."""
        if self.training_history is None:
            print("❌ No training history available. Train a model first.")
            return

        history = self.training_history.history

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')

        # Loss Plot
        axes[0, 0].plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy Plot
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy', color='green', linewidth=2)
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision Plot (check for existence)
        if 'precision' in history and 'val_precision' in history:
            axes[1, 0].plot(history['precision'], label='Training Precision', color='purple', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Validation Precision', color='brown', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off') # Hide if no data
            axes[1, 0].text(0.5, 0.5, 'Precision data not available', horizontalalignment='center', verticalalignment='center', transform=axes[1,0].transAxes)

        # Recall Plot (check for existence)
        if 'recall' in history and 'val_recall' in history:
            axes[1, 1].plot(history['recall'], label='Training Recall', color='cyan', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Validation Recall', color='magenta', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off') # Hide if no data
            axes[1, 1].text(0.5, 0.5, 'Recall data not available', horizontalalignment='center', verticalalignment='center', transform=axes[1,1].transAxes)


        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training plots saved to {save_path}")

    def metric(self, save_path='crm.png'):
        """Generates and plots confusion matrix and classification report for validation data."""
        if self.model is None or self.validation_data is None:
            print("❌ No model or validation data available. Train a model first.")
            return

        X_val, y_val = self.validation_data

        print("🔮 Generating predictions for validation data...")
        # Ensure X_val is correctly typed for prediction
        # model.predict expects a tf.Tensor or numpy array. It usually handles this fine
        # but explicit tf.constant ensures it if coming from another source.
        if not isinstance(X_val, tf.Tensor):
            X_val = tf.constant(X_val, dtype=tf.float32)
        
        # Ensure shape matches expected input for prediction
        # The input_dim should be set after the vectorizer is fitted.
        if self.input_dim is not None:
             X_val = tf.ensure_shape(X_val, (None, self.input_dim))

        y_pred_prob = self.model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')

        # Ensure y_val is a numpy array for confusion_matrix and classification_report
        y_val_np = y_val.numpy().flatten() if isinstance(y_val, tf.Tensor) else y_val.flatten()

        cm = confusion_matrix(y_val_np, y_pred)
        labels = self.label_encoder.classes_

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        cm_norm = confusion_matrix(y_val_np, y_pred, normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=labels, yticklabels=labels, ax=axes[0, 1])
        axes[0, 1].set_title('Normalized Confusion Matrix', fontweight='bold')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        report = classification_report(y_val_np, y_pred, target_names=labels, output_dict=True)
        axes[1, 0].axis('off') # Hide axis for text

        report_text = f"""
Classification Report:

                 precision    recall  f1-score   support

{labels[0]:<12} {report[labels[0]]['precision']:.3f}     {report[labels[0]]['recall']:.3f}     {report[labels[0]]['f1-score']:.3f}     {int(report[labels[0]]['support'])}
{labels[1]:<12} {report[labels[1]]['precision']:.3f}     {report[labels[1]]['recall']:.3f}     {report[labels[1]]['f1-score']:.3f}     {int(report[labels[1]]['support'])}

accuracy                         {report['accuracy']:.3f}     {int(report['macro avg']['support'])}
macro avg        {report['macro avg']['precision']:.3f}     {report['macro avg']['recall']:.3f}     {report['macro avg']['f1-score']:.3f}     {int(report['macro avg']['support'])}
weighted avg     {report['weighted avg']['precision']:.3f}     {report['weighted avg']['recall']:.3f}     {report['weighted avg']['f1-score']:.3f}     {int(report['weighted avg']['support'])}
        """

        axes[1, 0].text(0.1, 0.5, report_text, fontfamily='monospace',
                        fontsize=10, transform=axes[1, 0].transAxes, verticalalignment='center')
        axes[1, 0].set_title('Classification Report', fontweight='bold')

        axes[1, 1].hist(y_pred_prob, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Prediction Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Confusion matrix and metrics saved to {save_path}")

        print(f"\n📊 Model Performance Summary:")
        print(f"   Accuracy:  {report['accuracy']:.4f}")
        print(f"   Precision: {report['weighted avg']['precision']:.4f}")
        print(f"   Recall:    {report['weighted avg']['recall']:.4f}")
        print(f"   F1-Score:  {report['weighted avg']['f1-score']:.4f}")

    def load_data(self, train_file='train.txt', test_file='test.txt', val_file='val.txt'):
        """Loads data from specified text files into a pandas DataFrame."""
        print("Loading data...")

        def read_file(filename):
            urls = []
            labels = []
            try:
                with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue # Skip empty lines
                        
                        # Try splitting by tab first, then space
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            label, url = parts
                        else: # Try splitting by space
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                label, url = parts
                            else:
                                # If neither tab nor space works, skip or log
                                print(f"Warning: Could not parse line in {filename}: '{line}'")
                                continue
                        
                        labels.append(label.strip())
                        urls.append(url.strip())
                return urls, labels
            except FileNotFoundError:
                print(f"File {filename} not found!")
                return [], []
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                return [], []

        train_urls, train_labels = read_file(train_file)
        test_urls, test_labels = read_file(test_file)
        val_urls, val_labels = read_file(val_file)

        all_urls = train_urls + test_urls + val_urls
        all_labels = train_labels + test_labels + val_labels

        print(f"Loaded {len(all_urls)} URLs total")
        print(f"Train: {len(train_urls)}, Test: {len(test_urls)}, Val: {len(val_urls)}")

        df = pd.DataFrame({
            'url': all_urls,
            'label': all_labels
        })

        df = df.dropna().reset_index(drop=True)
        df = df[df['url'].str.len() > 0].reset_index(drop=True)

        print(f"After cleaning: {len(df)} URLs")
        print("Label distribution:")
        print(df['label'].value_counts())

        return df

    def preprocess_data(self, df):
        """Initial preprocessing to fit vectorizer and label encoder."""
        print("Preprocessing data to fit vectorizer and encoder...")

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df['label']) # Fit once on all data for consistency

        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                analyzer='char',
                lowercase=True,
                max_df=0.95,
                min_df=2
            )
            print("Fitting TF-IDF vectorizer on all URLs for feature space definition...")
            self.vectorizer.fit(df['url'])
            
        dummy_url_features = self.extract_url_features("http://dummy.com")
        num_url_features = len(dummy_url_features)
        num_tfidf_features = len(self.vectorizer.get_feature_names_out())
        
        self.input_dim = num_url_features + num_tfidf_features
        print(f"Determined total input dimension: {self.input_dim}")
        
        if self.input_dim == 0:
            raise ValueError("Calculated input dimension is 0. Check feature extraction or TF-IDF parameters.")

    @tf.function # Decorate with tf.function for performance
    def _extract_and_transform_single(self, url_tensor, label_tensor):
        """
        TensorFlow-compatible function to extract features and transform labels
        for a single URL-label pair from the dataset map.
        """
        url = url_tensor.numpy().decode('utf-8') # Decode byte string to Python string

        # Extract numerical features
        url_features_np = np.array(self.extract_url_features(url), dtype=np.float32)

        # Transform URL using the fitted TF-IDF vectorizer
        # vectorizer.transform expects a list of strings
        tfidf_features_np = self.vectorizer.transform([url]).toarray().astype(np.float32).squeeze()

        # Concatenate features
        X_np = np.hstack([url_features_np, tfidf_features_np]).astype(np.float32)
        
        # Transform label (assuming label_tensor is already binary 0/1 for now)
        # Convert label back to string for encoder, then transform
        label_str = label_tensor.numpy().decode('utf-8') # Assuming label is also string
        y_np = np.array(self.label_encoder.transform([label_str]), dtype=np.float32).reshape(-1, 1)

        return X_np, y_np

    def _create_tf_dataset(self, df_subset, shuffle=True):
        """Creates a tf.data.Dataset from a pandas DataFrame subset."""
        # It's generally better to perform feature extraction in numpy first,
        # then create a dataset from the prepared numpy arrays,
        # as TF-IDF is a scikit-learn object not directly TF compatible.

        print(f"Preparing {len(df_subset)} samples for tf.data.Dataset...")
        
        # 1. Extract all features and labels into numpy arrays
        # This part still needs to be done in Python/NumPy, as TF-IDF and
        # custom URL feature extraction are not easily vectorized with TF operations.
        X_data_list = []
        y_data_list = []

        # Process in chunks to avoid memory issues for very large DFs
        chunk_size = 10000 
        for i in range(0, len(df_subset), chunk_size):
            chunk_df = df_subset.iloc[i:i+chunk_size]
            
            # Extract URL features
            url_features_chunk = np.array([self.extract_url_features(url) for url in chunk_df['url']], dtype=np.float32)
            
            # Extract TF-IDF features
            tfidf_features_chunk = self.vectorizer.transform(chunk_df['url']).toarray().astype(np.float32)
            
            # Combine them
            X_chunk = np.hstack([url_features_chunk, tfidf_features_chunk]).astype(np.float32)
            
            # Encode labels
            y_chunk = self.label_encoder.transform(chunk_df['label']).astype(np.float32).reshape(-1, 1)
            
            X_data_list.append(X_chunk)
            y_data_list.append(y_chunk)
            
            if i % (chunk_size * 5) == 0 and i > 0:
                print(f"Processed {min(i + chunk_size, len(df_subset))}/{len(df_subset)} samples for dataset creation.")

        X_data = np.vstack(X_data_list)
        y_data = np.vstack(y_data_list)

        print(f"Created numpy arrays for dataset. X_data.shape: {X_data.shape}, y_data.shape: {y_data.shape}")
        
        # Ensure correct input_dim if not set yet (e.g., in use() after load)
        if self.input_dim is None:
            self.input_dim = X_data.shape[1]

        dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X_data) + 100) # Large buffer for better shuffling

        return dataset

    def create_model(self, input_dim):
        """Creates the Keras Sequential model."""
        model = keras.Sequential([
            # Explicit Input layer with dtype
            layers.Input(shape=(input_dim,), dtype=tf.float32), 
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid') # Binary classification output
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            # Using Keras Metric objects explicitly here
        )
        return model

    def train_model(self, df, epochs=50, batch_size=32, validation_split=0.2, sample_size=100000):
        """Trains the model using tf.data.Dataset."""
        print("Starting model training with tf.data.Dataset...")

        if len(df) > sample_size:
            print(f"Dataset too large ({len(df)} samples). Sampling {sample_size} samples for training...")
            df_sampled = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
            ).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Sampled dataset size: {len(df_sampled)}")
            print("Sampled label distribution:")
            print(df_sampled['label'].value_counts())
        else:
            df_sampled = df

        # Step 1: Preprocess the entire sampled DataFrame to fit vectorizer and encoder
        # This will also set self.input_dim
        self.preprocess_data(df_sampled) 

        # Step 2: Split DataFrame indices for train/validation
        train_indices, val_indices = train_test_split(
            df_sampled.index, test_size=validation_split, random_state=42, stratify=df_sampled['label']
        )
        
        train_df = df_sampled.iloc[train_indices].reset_index(drop=True)
        val_df = df_sampled.iloc[val_indices].reset_index(drop=True)

        # Step 3: Create tf.data.Dataset objects
        # The _create_tf_dataset method now extracts features and labels fully in numpy first.
        train_dataset = self._create_tf_dataset(train_df, shuffle=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = self._create_tf_dataset(val_df, shuffle=False).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Create the model using the determined input_dim
        self.model = self.create_model(self.input_dim)
        print("Model architecture:")
        self.model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]

        print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")
        try:
            history = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=1
            )
            self.training_history = history
        except Exception as e:
            print(f"\nFATAL ERROR DURING MODEL FIT (using tf.data.Dataset): {e}")
            print("This could indicate an issue with the tf.data.Dataset pipeline or an internal TF error.")
            raise

        print("Preparing validation data for metrics...")
        # To get a single tensor for metrics, iterate through the validation dataset
        # and stack them. This is for the final metrics calculation after training.
        X_val_list = []
        y_val_list = []
        for x_batch, y_batch in val_dataset.unbatch().take(min(5000, len(val_df))): # Take a subset
            X_val_list.append(x_batch)
            y_val_list.append(y_batch)
        
        # Concatenate into single Tensors
        X_val_final = tf.concat(X_val_list, axis=0)
        y_val_final = tf.concat(y_val_list, axis=0)

        self.validation_data = (X_val_final, y_val_final)

        print("\nEvaluating on validation dataset subset...")
        # Model evaluate method directly accepts tf.data.Dataset as well
        eval_results = self.model.evaluate(val_dataset, verbose=0)
        
        # Ensure eval_results has enough elements for all metrics
        # The order is usually loss, then metrics in the order they are compiled
        if len(eval_results) >= 4:
            test_loss, test_acc, test_prec, test_rec = eval_results[0], eval_results[1], eval_results[2], eval_results[3]
        else:
            print("Warning: Not enough evaluation results for all metrics (loss, accuracy, precision, recall).")
            print(f"Raw evaluation results: {eval_results}")
            # Fallback if metrics are missing (e.g., if precision/recall weren't computed)
            test_loss, test_acc = eval_results[0], eval_results[1]
            test_prec, test_rec = 0.0, 0.0 # Default to 0 if not explicitly returned
        
        f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test F1-Score: {f1_score:.4f}")

        return history

    def save_model(self, model_path='phishing_model.h5', vectorizer_path='vectorizer.pkl',
                   encoder_path='label_encoder.pkl'):
        """Saves the trained model and preprocessing objects."""
        if self.model is None:
            print("No model to save! Train a model first.")
            return

        print("Saving model and preprocessing objects...")

        try:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving full model: {e}. Attempting to save weights only.")
            # Fallback to save weights if full model save fails
            # Need a dummy model to load weights later
            self.model.save_weights(model_path.replace('.h5', '_weights.h5'))
            print(f"Model weights saved to {model_path.replace('.h5', '_weights.h5')}")

        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"Vectorizer saved to {vectorizer_path}")
        except Exception as e:
            print(f"Error saving vectorizer: {e}")

        try:
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"Label encoder saved to {encoder_path}")
        except Exception as e:
            print(f"Error saving label encoder: {e}")

        print("All components save attempts complete!")

    def load_saved_model(self, model_path='phishing_model.h5', vectorizer_path='vectorizer.pkl',
                         encoder_path='label_encoder.pkl'):
        """Loads a previously saved model and preprocessing objects."""
        print("Loading saved model...")

        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            # After loading, set the input_dim from the loaded model
            self.input_dim = self.model.input_shape[1]
            print(f"Loaded model input dimension: {self.input_dim}")

        except Exception as e_model:
            print(f"Error loading full model from {model_path}: {e_model}")
            weights_path = model_path.replace('.h5', '_weights.h5')
            if os.path.exists(weights_path):
                print(f"Attempting to load weights from {weights_path}. You might need to manually recreate the model structure first.")
                # To load weights, you need to create the model architecture first,
                # which requires input_dim. This is a weakness if only weights are saved.
                # A more robust solution would save input_dim alongside objects.
                # For now, we'll assume the full model is saved or input_dim is derived.
                
                # To make this truly robust, you'd need to load vectorizer/encoder first
                # to get input_dim, then build model, then load weights.
                try:
                    with open(vectorizer_path, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    dummy_url_features = self.extract_url_features("http://dummy.com")
                    num_url_features = len(dummy_url_features)
                    num_tfidf_features = len(self.vectorizer.get_feature_names_out())
                    self.input_dim = num_url_features + num_tfidf_features
                    self.model = self.create_model(self.input_dim)
                    self.model.load_weights(weights_path)
                    print(f"Model weights loaded from {weights_path}")
                except Exception as e_weights:
                     print(f"Failed to load weights even after recreating model: {e_weights}")
                     return False
            else:
                print("No model or weights file found for loading.")
                return False

        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"Vectorizer loaded from {vectorizer_path}")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            return False

        try:
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"Label encoder loaded from {encoder_path}")
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            return False

        print("Model and preprocessing objects loaded successfully!")
        return True

    def use(self, url):
        """Makes a prediction for a single URL."""
        if self.model is None or self.vectorizer is None or self.label_encoder is None or self.input_dim is None:
            print("Model or preprocessing objects not loaded! Load a saved model or train a new one.")
            return None

        # Prepare URL features
        url_features_np = np.array([self.extract_url_features(url)], dtype=np.float32)

        # Prepare TF-IDF features
        tfidf_features_np = self.vectorizer.transform([url]).toarray().astype(np.float32)

        # Concatenate all features
        X_combined_np = np.hstack([url_features_np, tfidf_features_np])

        # Ensure the input shape matches the model's expected input shape
        if X_combined_np.shape[1] != self.input_dim:
            print(f"Error: Input feature dimension mismatch for prediction. Expected {self.input_dim}, got {X_combined_np.shape[1]}")
            return None

        # Convert to TensorFlow tensor
        X_inference = tf.constant(X_combined_np, dtype=tf.float32)

        # Make prediction
        prediction_prob = self.model.predict(X_inference, verbose=0)[0][0]
        prediction_binary = int(prediction_prob > 0.5)

        prediction_label = self.label_encoder.inverse_transform([prediction_binary])[0]

        return {
            'url': url,
            'prediction': prediction_label,
            'confidence': float(prediction_prob if prediction_label == self.label_encoder.classes_[1] else 1 - prediction_prob)
            # Assuming 'phishing' is the positive class (index 1) after encoding
        }


if __name__ == "__main__":
    detector = PhishingURLDetector()

    # Load data from your files
    df = detector.load_data('data/big_dataset/train.txt', 'data/big_dataset/test.txt', 'data/big_dataset/val.txt')

    if not df.empty:
        try:
            print("\n🚀 Starting model training with tf.data.Dataset (most robust approach)...")
            print("⚠️  Press Ctrl+C anytime to save model and exit gracefully")

            # train_model now internally handles fitting vectorizer and label_encoder
            history = detector.train_model(df, epochs=20, batch_size=64, sample_size=100000)

            print("\n💾 Saving model...")
            detector.save_model()

            print("\n📊 Generating training plots...")
            detector.plot('training_plot.png')

            print("📈 Generating confusion matrix and metrics...")
            detector.metric('confusion_matrix.png')

            test_urls = [
                "http://www.bartekbitner.pl/libraries/fof/-/din7",
                "https://eheadspace.org.au/headspace-centres/murray-bridge/",
                "https://www.google.com",
                "http://malicious-site-12345.com/fake-login"
            ]

            print("\n🔮 Testing predictions on example URLs:")
            for url in test_urls:
                result = detector.use(url)
                if result:
                    print(f"URL: {url}")
                    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
                    print("-" * 80)
                else:
                    print(f"Could not get prediction for URL: {url}")
                    print("-" * 80)

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user!")
        except Exception as e:
            print(f"\nFATAL UNHANDLED ERROR DURING EXECUTION: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for deeper insights

    else:
        print("❌ No data loaded or data frame is empty. Please check your data files and their content.")