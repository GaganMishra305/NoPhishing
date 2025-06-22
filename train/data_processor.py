import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from urllib.parse import urlparse
from scipy.sparse import hstack

class DataProcessor:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2), # Consider both single characters and character pairs
            analyzer='char',    # Analyze character by character
            lowercase=True,     # Convert text to lowercase
            max_df=0.95,        # Ignore terms that appear in more than 95% of documents
            min_df=2            # Ignore terms that appear in less than 2 documents
        )
        self.label_encoder = LabelEncoder()
        self.input_dim = None # To store the final input dimension after preprocessing

    def extract_url_features(self, url):
        features = []

        # Length-based features
        features.append(len(url))               # Total length of the URL
        features.append(url.count('.'))         # Number of dots
        features.append(url.count('/'))         # Number of slashes
        features.append(url.count('-'))         # Number of hyphens
        features.append(url.count('_'))         # Number of underscores
        features.append(url.count('?'))         # Presence of query parameters
        features.append(url.count('='))         # Number of equals signs
        features.append(url.count('&'))         # Number of ampersands

        # Presence of common indicators
        features.append(1 if 'https' in url else 0) # Is it HTTPS?
        features.append(1 if any(char.isdigit() for char in url) else 0) # Contains digits?
        
        # Presence of IP address in hostname (e.g., http://192.168.1.1/malicious)
        features.append(1 if re.search(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', url) else 0)

        # Parsed URL features
        try:
            parsed = urlparse(url)
            domain = parsed.netloc # Get the network location (domain/IP)
            features.append(len(domain)) # Length of the domain
            features.append(domain.count('.')) # Number of dots in the domain
        except:
            # Default to 0 if URL parsing fails for any reason
            features.append(0)
            features.append(0)
        
        return features

    def load_data(self, train_file, test_file, val_file , max_rows = 100000):
        print("Loading data...")

        def read_file(filename, mx_rows):
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
                                # If neither tab nor space works, skip or log a warning
                                print(f"Warning: Could not parse line in {filename}: '{line}'")
                                continue
                        
                        labels.append(label.strip())
                        urls.append(url.strip())
                return urls[:mx_rows], labels[:mx_rows]
            except FileNotFoundError:
                print(f"File {filename} not found!")
                return [], []
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                return [], []

        train_urls, train_labels = read_file(train_file, mx_rows=int(max_rows*0.7))
        test_urls, test_labels = read_file(test_file, mx_rows=int(max_rows*0.2))
        val_urls, val_labels = read_file(val_file, mx_rows=int(max_rows*0.1))

        all_urls = train_urls + test_urls + val_urls
        all_labels = all_labels = [label.lower() for label in train_labels + test_labels + val_labels]


        print(f"Loaded {len(all_urls)} URLs total")
        print(f"Train: {len(train_urls)}, Test: {len(test_urls)}, Val: {len(val_urls)}")

        df = pd.DataFrame({
            'url': all_urls,
            'label': all_labels
        })

        # Remove rows with missing values or empty URLs
        df = df.dropna().reset_index(drop=True)
        df = df[df['url'].str.len() > 0].reset_index(drop=True)

        print(f"After initial cleaning: {len(df)} URLs")
        print("Label distribution:")
        print(df['label'].value_counts())

        return df

    def preprocess_and_transform(self, df_urls, df_labels=None, fit_vectorizer=True, fit_encoder=True):
        print("Preprocessing and transforming data...")

        # Fit LabelEncoder if required
        if fit_encoder and df_labels is not None:
            print("Fitting LabelEncoder...")
            self.label_encoder.fit(df_labels)
            print(f"Detected classes: {self.label_encoder.classes_}")
        elif fit_encoder and df_labels is None:
            raise ValueError("Labels must be provided to fit the LabelEncoder.")
            
        # Fit TfidfVectorizer if required
        if fit_vectorizer:
            print("Fitting TF-IDF vectorizer...")
            self.vectorizer.fit(df_urls)
            print(f"TF-IDF features count: {len(self.vectorizer.get_feature_names_out())}")
        
        # Extract numerical features for all URLs
        print("Extracting numerical URL features...")
        url_features = np.array([self.extract_url_features(url) for url in df_urls], dtype=np.float32)

        # Transform URLs using the fitted TF-IDF vectorizer
        print("Transforming URLs with TF-IDF vectorizer...")
        tfidf_features = self.vectorizer.transform(df_urls).toarray().astype(np.float32)

        # Concatenate numerical and TF-IDF features
        X_combined = np.hstack([url_features, tfidf_features])
        self.input_dim = X_combined.shape[1]
        print(f"Combined feature shape: {X_combined.shape}")
        print(f"Total input dimension set to: {self.input_dim}")

        y_encoded = None
        if df_labels is not None:
            # Transform labels using the fitted LabelEncoder
            print("Encoding labels...")
            y_encoded = self.label_encoder.transform(df_labels).astype(np.float32).reshape(-1, 1)
            print(f"Encoded labels shape: {y_encoded.shape}")

        return X_combined, y_encoded

    def create_tf_dataset(self, X, y, shuffle=True, batch_size=32):
        print(f"Creating tf.data.Dataset for {len(X)} samples (shuffle={shuffle}, batch_size={batch_size})...")
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            # Add a buffer for shuffling for better randomness
            dataset = dataset.shuffle(buffer_size=len(X) + 100) 

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Prefetch for performance
        print("Dataset created.")
        return dataset

    def get_feature_names(self):
        if self.vectorizer:
            return self.vectorizer.get_feature_names_out()
        return []
