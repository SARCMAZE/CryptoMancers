import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load saved model, label encoder, and TF-IDF vectorizers
with open("asymm
etric_key_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

with open("tfidf_vectorizer_pub.pkl", "rb") as pub_tfidf_file:
    tfidf_vectorizer_pub = pickle.load(pub_tfidf_file)

with open("tfidf_vectorizer_pri.pkl", "rb") as pri_tfidf_file:
    tfidf_vectorizer_pri = pickle.load(pri_tfidf_file)

# StandardScaler for numerical features
scaler = StandardScaler()

# Function to calculate entropy
def calculate_entropy(text):
    if not text:  # Handle empty strings
        return 0
    probability_distribution = np.bincount(bytearray(text.encode("utf-8"))) / len(text)
    from scipy.stats import entropy
    return entropy(probability_distribution, base=2)

# Function to preprocess inputs and predict
def predict_encryption_algorithm(plaintext, ciphertext, public_key, private_key):
    # Compute features
    plaintext_length = len(plaintext)
    ciphertext_length = len(ciphertext)
    ciphertext_entropy = calculate_entropy(ciphertext)
    
    # TF-IDF transformations
    public_key_tfidf = tfidf_vectorizer_pub.transform([public_key]).toarray()
    private_key_tfidf = tfidf_vectorizer_pri.transform([private_key]).toarray()

    # Combine all features
    input_features = np.hstack([
        [[plaintext_length, ciphertext_length, ciphertext_entropy]],
        public_key_tfidf,
        private_key_tfidf,
    ])
    
    # Standardize numerical features
    input_features[:, :3] = scaler.fit_transform(input_features[:, :3])

    # Make prediction
    prediction = rf_model.predict(input_features)
    predicted_algorithm = label_encoder.inverse_transform(prediction)
    return predicted_algorithm[0]

# Use case: Input from user
if __name__ == "__main__":
    print("Enter the following details for prediction:")
    plaintext = input("Plaintext: ").strip()
    ciphertext = input("Ciphertext: ").strip()
    public_key = input("Public Key: ").strip()
    private_key = input("Private Key: ").strip()

    # Perform prediction
    predicted_algorithm = predict_encryption_algorithm(plaintext, ciphertext, public_key, private_key)
    print(f"Predicted Encryption Algorithm: {predicted_algorithm}")
