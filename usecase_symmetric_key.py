import numpy as np
import pickle
from scipy.stats import entropy, skew, kurtosis

# Load the optimized model
model_filename = '/content/symmetric_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the label encoder (optional, if required for mapping)
label_encoder_filename = '/content/symmetric_label_encoder_k.pkl'
with open(label_encoder_filename, 'rb') as file:
    label_encoder = pickle.load(file)

# Feature extraction function
def extract_features(plaintext, ciphertext, key):
    def compute_features(data):
        byte_array = np.array(list(data.encode()), dtype=np.uint8)
        mean = np.mean(byte_array)
        variance = np.var(byte_array)
        entropy_value = entropy(np.bincount(byte_array, minlength=256) / len(byte_array))
        skewness = skew(byte_array)
        kurt = kurtosis(byte_array)
        return [mean, variance, entropy_value, skewness, kurt]

    plaintext_features = compute_features(plaintext)
    ciphertext_features = compute_features(ciphertext)
    key_features = compute_features(key)
    return plaintext_features + ciphertext_features + key_features

# Function to predict the encryption algorithm
def predict_algorithm(plaintext, ciphertext, key):
    # Extract features from input
    features = np.array([extract_features(plaintext, ciphertext, key)])
    # Predict using the loaded model
    prediction = model.predict(features)
    # Decode the label back to the algorithm name
    predicted_algorithm = label_encoder.inverse_transform(prediction)[0]
    return predicted_algorithm

# Interactive user input
def main():
    print("=== Encryption Algorithm Prediction ===")
    plaintext = input("Enter plaintext: ")
    ciphertext = input("Enter ciphertext: ")
    key = input("Enter key: ")

    try:
        # Predict the algorithm
        algorithm = predict_algorithm(plaintext, ciphertext, key)
        print(f"\nPredicted Encryption Algorithm: {algorithm}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
