import os
import signal
import sys
import pickle
from datetime import datetime
import warnings

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Suppress warnings that might clutter the output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# Import modules from our refactored structure
from data_processor import DataProcessor
from model_architectures import ModelArchitectures
from plot_utils import PlotUtils
from pipeline import PhishingPredictionPipeline 

class PhishingURLDetector:
    def __init__(self, max_features=5000):
        self.data_processor = DataProcessor(max_features=max_features)
        self.model_builder = ModelArchitectures()
        self.plot_utils = PlotUtils()

        self.model = None # Current active model (can be Keras or sklearn)
        self.training_history = None # Stores the history object from model.fit (for Keras models)
        self.validation_data_for_metrics = None # Stores X_val, y_val as NumPy arrays for metric calculation

        # Dictionary to store performance metrics for all trained models for comparison
        self.all_models_performance = {}

        self.interrupt_handler_set = False
        self.setup_interrupt_handler() # Set up graceful exit on Ctrl+C

    def setup_interrupt_handler(self):
        if not self.interrupt_handler_set:
            def signal_handler(sig, frame):
                print('\n⚠️  Keyboard interrupt detected!')
                if self.model is not None:
                    print('💾 Attempting to save current model and preprocessors as a pipeline before exit...')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    try:
                        # Instantiate pipeline object
                        pipeline = PhishingPredictionPipeline(
                            self.model,
                            self.data_processor.vectorizer,
                            self.data_processor.label_encoder,
                            self.data_processor.input_dim
                        )
                        default_model_dir = "interrupted_saved_models"
                        os.makedirs(default_model_dir, exist_ok=True)
                        pipeline_path = os.path.join(default_model_dir, f'interrupted_pipeline_{timestamp}.pkl')
                        
                        with open(pipeline_path, 'wb') as f:
                            pickle.dump(pipeline, f)
                        print(f'✅ Pipeline saved successfully to {pipeline_path}!')
                    except Exception as e:
                        print(f'❌ Error saving pipeline during interruption: {e}')
                else:
                    print('❌ No model to save during interruption.')
                print('👋 Exiting gracefully...')
                sys.exit(0) # Exit the program

            # Register the signal handler for SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, signal_handler)
            self.interrupt_handler_set = True
            print("Interrupt handler set up. Press Ctrl+C to save and exit during training.")

    # Removed save_preprocessors as it's now encapsulated in pipeline save

    def train_model(self, df, architecture, epochs=50, batch_size=32, validation_split=0.2, sample_size=100000):
        print(f"\n🚀 Starting training for {architecture} model...")

        # If the dataset is too large, sample a subset to manage memory
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
        # This also sets self.data_processor.input_dim
        X_all, y_all = self.data_processor.preprocess_and_transform(
            df_sampled['url'], df_sampled['label'], 
            fit_vectorizer=True, fit_encoder=True
        )

        # Step 2: Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=validation_split, random_state=42, stratify=y_all
        )
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Store validation data for later metric calculation (e.g., confusion matrix)
        self.validation_data_for_metrics = (X_val, y_val)

        # Step 3: Build the model based on the chosen architecture
        self.model = self.model_builder.build_model(architecture, self.data_processor.input_dim)
        
        current_model_performance = {}

        if isinstance(self.model, keras.Model):
            # Keras Model Training
            print(f"\nModel Architecture Summary for {architecture}:")
            self.model.summary()

            # Create TensorFlow Datasets from numpy arrays
            train_dataset = self.data_processor.create_tf_dataset(X_train, y_train, shuffle=True, batch_size=batch_size)
            val_dataset = self.data_processor.create_tf_dataset(X_val, y_val, shuffle=False, batch_size=batch_size)

            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
            ]

            print(f"\nTraining {architecture} model for {epochs} epochs with batch size {batch_size}...")
            try:
                history = self.model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    verbose=1
                )
                self.training_history = history
                print(f"\n{architecture} model training completed.")
            except Exception as e:
                print(f"\nFATAL ERROR DURING {architecture} MODEL TRAINING: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Evaluate the model
            print(f"\nEvaluating {architecture} model on validation data...")
            eval_results = self.model.evaluate(val_dataset, verbose=0)
            
            # Ensure eval_results has enough elements based on compiled metrics
            if len(eval_results) >= 4:
                val_loss, val_acc, val_prec, val_rec = eval_results[0], eval_results[1], eval_results[2], eval_results[3]
            else:
                print(f"Warning: Not all metrics (precision, recall) were returned for {architecture} from model evaluation.")
                val_loss, val_acc = eval_results[0], eval_results[1]
                val_prec, val_rec = 0.0, 0.0 # Default if not available
            
            val_f1 = 2 * (val_prec * val_rec) / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0

            print(f"Validation Loss ({architecture}): {val_loss:.4f}")
            print(f"Validation Accuracy ({architecture}): {val_acc:.4f}")
            print(f"Validation Precision ({architecture}): {val_prec:.4f}")
            print(f"Validation Recall ({architecture}): {val_rec:.4f}")
            print(f"Validation F1-Score ({architecture}): {val_f1:.4f}")

            current_model_performance = {
                'accuracy': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1_score': val_f1,
                'loss': val_loss # Storing loss for Keras models
            }

        else: # Sklearn-like model (e.g., XGBoost)
            # Reshape y_train and y_val for sklearn models if they are (N, 1)
            y_train_flat = y_train.flatten()
            y_val_flat = y_val.flatten()

            print(f"\nTraining {architecture} model...")
            self.model.fit(X_train, y_train_flat)

            print(f"\nEvaluating {architecture} model on validation data...")
            y_pred_val = self.model.predict(X_val)
            y_pred_prob_val = self.model.predict_proba(X_val)[:, 1] # Probability of positive class

            val_acc = accuracy_score(y_val_flat, y_pred_val)
            val_prec = precision_score(y_val_flat, y_pred_val)
            val_rec = recall_score(y_val_flat, y_pred_val)
            val_f1 = f1_score(y_val_flat, y_pred_val)

            print(f"Validation Accuracy ({architecture}): {val_acc:.4f}")
            print(f"Validation Precision ({architecture}): {val_prec:.4f}")
            print(f"Validation Recall ({architecture}): {val_rec:.4f}")
            print(f"Validation F1-Score ({architecture}): {val_f1:.4f}")
            
            # Calculate a "loss" metric if meaningful for non-Keras (e.g., log_loss)
            # For simplicity, we might not track 'loss' for non-Keras models in comparison plot
            current_model_performance = {
                'accuracy': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1_score': val_f1,
                'loss': None # No direct 'loss' equivalent from fit history for sklearn models
            }

        self.all_models_performance[architecture] = current_model_performance
        return current_model_performance


    def save_pipeline(self, architecture_name, model_dir='saved_models'):
        if self.model is None:
            print(f"No {architecture_name} model to save! Train it first.")
            return

        # Create architecture-specific subdirectory for the pipeline
        arch_pipeline_dir = os.path.join(model_dir, architecture_name.lower())
        os.makedirs(arch_pipeline_dir, exist_ok=True)
        
        pipeline_path = os.path.join(arch_pipeline_dir, f'{architecture_name.lower()}_pipeline.pkl')
        
        print(f"Saving complete {architecture_name} pipeline to {pipeline_path}...")

        model_to_save_in_pipeline = self.model
        if isinstance(self.model, keras.Model):
            temp_keras_model_path = os.path.join(arch_pipeline_dir, f'_{architecture_name.lower()}_temp_full_model.h5')
            try:
                self.model.save(temp_keras_model_path)
                model_to_save_in_pipeline = keras.models.load_model(temp_keras_model_path)
                print(f"  Keras model temporarily saved and reloaded for pickling stability.")
            except Exception as e:
                print(f"  Warning: Could not temporarily save/load Keras model for pickling: {e}. "
                      f"Attempting to pickle raw Keras model. This might lead to issues.")
            finally:
                if os.path.exists(temp_keras_model_path):
                    os.remove(temp_keras_model_path) # Clean up temporary file

        try:
            pipeline = PhishingPredictionPipeline(
                model_to_save_in_pipeline,
                self.data_processor.vectorizer,
                self.data_processor.label_encoder,
                self.data_processor.input_dim
            )
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline, f)
            print(f"✅ Complete {architecture_name} pipeline saved to {pipeline_path}")
        except Exception as e:
            print(f"❌ Error saving complete {architecture_name} pipeline: {e}")

        print(f"{architecture_name} pipeline save attempt complete!")

    def plot_training_history(self, architecture_name, save_path_base='training_history'):
        if self.training_history is None:
            print(f"❌ No training history available for {architecture_name}. (Only Keras models have history).")
            return
        # Ensure the save path includes architecture name
        save_path = os.path.join(save_path_base, f'{architecture_name.lower()}_training_history.png')
        self.plot_utils.plot_training_history(self.training_history, save_path)

    def generate_metrics_plots(self, architecture_name, save_path_base='performance_metrics'):
        if self.validation_data_for_metrics is None or len(self.validation_data_for_metrics) != 2:
            print("❌ Validation data for metrics is not available. Train the model first.")
            return

        X_val, y_val = self.validation_data_for_metrics

        if self.model is None:
            print("❌ Model not loaded to generate predictions for metrics.")
            return

        y_pred_prob = None
        
        if isinstance(self.model, keras.Model):
            # For Keras models, potentially reshape if it's CNN/RNN
            X_val_tensor = tf.constant(X_val, dtype=tf.float32)
            if self.model.name.startswith("PhishingURL_CNN") or self.model.name.startswith("PhishingURL_RNN"):
                X_val_tensor = tf.reshape(X_val_tensor, (X_val_tensor.shape[0], X_val_tensor.shape[1], 1))
            y_pred_prob = self.model.predict(X_val_tensor, verbose=0).flatten()
        else: # Scikit-learn type model
            y_pred_prob = self.model.predict_proba(X_val)[:, 1].flatten()

        labels = self.data_processor.label_encoder.classes_
        
        # Ensure the save path includes architecture name
        save_path = os.path.join(save_path_base, f'{architecture_name.lower()}_performance_metrics.png')
        self.plot_utils.plot_confusion_matrix_and_report(y_val, y_pred_prob, labels.tolist(), save_path)

    def plot_all_models_comparison(self, save_path='model_comparison_summary.png'):
        self.plot_utils.plot_model_comparison(self.all_models_performance, save_path)
