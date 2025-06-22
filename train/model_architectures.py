import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from xgboost import XGBClassifier # Using XGBoost for a non-deep learning model
import numpy as np

class ModelArchitectures:
    def __init__(self):
        pass

    def build_ann_model(self, input_dim):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,), dtype=tf.float32, name="input_features_ann"), 
            
            layers.Dense(512, activation='relu', name="dense_1"),
            layers.Dropout(0.3, name="dropout_1"),
            
            layers.Dense(256, activation='relu', name="dense_2"),
            layers.Dropout(0.3, name="dropout_2"),
            
            layers.Dense(128, activation='relu', name="dense_3"),
            layers.Dropout(0.2, name="dropout_3"),
            
            layers.Dense(64, activation='relu', name="dense_4"),
            layers.Dropout(0.2, name="dropout_4"),
            
            layers.Dense(32, activation='relu', name="dense_5"),
            
            layers.Dense(1, activation='sigmoid', name="output_prediction_ann") 
        ], name="PhishingURL_ANN_Classifier")

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')       
            ]
        )
        return model

    def build_cnn_model(self, input_dim):
        model = keras.Sequential([
            # Input layer for the flat feature vector
            layers.Input(shape=(input_dim,), dtype=tf.float32, name="input_features_cnn"),
            # Reshape the input to be (input_dim, 1) to treat each feature as a timestep with 1 dimension.
            # This allows Conv1D layers to sweep across the numerical features.
            layers.Reshape((input_dim, 1), name="reshape_for_cnn"),
            
            # Adapted from cnn_complex in dl_models.py
            layers.Conv1D(filters=128, kernel_size=3, activation='tanh', padding='same', name="conv1d_1"),
            layers.MaxPooling1D(pool_size=3, name="max_pool1d_1"),
            layers.Dropout(0.2, name="dropout_cnn_1"), # Added dropout for regularization
            
            layers.Conv1D(filters=128, kernel_size=7, activation='tanh', padding='same', name="conv1d_2"),
            layers.Dropout(0.2, name="dropout_cnn_2"),
            
            layers.Conv1D(filters=128, kernel_size=5, activation='tanh', padding='same', name="conv1d_3"),
            layers.Dropout(0.2, name="dropout_cnn_3"),
            
            layers.Conv1D(filters=128, kernel_size=3, activation='tanh', padding='same', name="conv1d_4"),
            layers.MaxPooling1D(pool_size=3, name="max_pool1d_2"),
            layers.Dropout(0.2, name="dropout_cnn_4"),
            
            layers.Conv1D(filters=128, kernel_size=5, activation='tanh', padding='same', name="conv1d_5"),
            layers.Dropout(0.2, name="dropout_cnn_5"),
            
            layers.Conv1D(filters=128, kernel_size=3, activation='tanh', padding='same', name="conv1d_6"),
            layers.MaxPooling1D(pool_size=3, name="max_pool1d_3"),
            layers.Dropout(0.2, name="dropout_cnn_6"),
            
            layers.Flatten(name="flatten_cnn"),
            
            layers.Dense(1, activation='sigmoid', name="output_prediction_cnn")
        ], name="PhishingURL_CNN_Classifier")

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')       
            ]
        )
        return model

    def build_rnn_model(self, input_dim):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,), dtype=tf.float32, name="input_features_rnn"),
            # Reshape for LSTM: (input_dim, 1) means 'input_dim' timesteps, each with 1 feature.
            layers.Reshape((input_dim, 1), name="reshape_for_rnn"),
            
            # Adapted from brnn_complex in dl_models.py
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bidirectional_lstm_1"),
            layers.Dropout(0.3, name="dropout_rnn_1"), # Added dropout
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bidirectional_lstm_2"),
            layers.Dropout(0.3, name="dropout_rnn_2"),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bidirectional_lstm_3"),
            layers.Dropout(0.3, name="dropout_rnn_3"),

            layers.Bidirectional(layers.LSTM(128), name="bidirectional_lstm_4"), # Last LSTM, no return_sequences
            layers.Dropout(0.3, name="dropout_rnn_4"),
            
            layers.Dense(1, activation='sigmoid', name="output_prediction_rnn")
        ], name="PhishingURL_RNN_Classifier")

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')       
            ]
        )
        return model

    def build_xgboost_model(self):
        model = XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss',       
            use_label_encoder=False,     
            n_estimators=100,            
            learning_rate=0.1,           
            max_depth=5,                 
            subsample=0.8,               
            colsample_bytree=0.8,        
            random_state=42              
        )
        return model

    def build_model(self, architecture_name, input_dim):
        if architecture_name.upper() == 'ANN':
            return self.build_ann_model(input_dim)
        elif architecture_name.upper() == 'CNN':
            return self.build_cnn_model(input_dim)
        elif architecture_name.upper() == 'RNN':
            return self.build_rnn_model(input_dim)
        elif architecture_name.upper() == 'XGBOOST':
            return self.build_xgboost_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture_name}. Choose from 'ANN', 'CNN', 'RNN', 'XGBoost'.")
    
    # can be used for defining custom models
    def build_custom_model(self):
        pass

