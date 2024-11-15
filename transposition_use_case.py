import pandas as pd
from joblib import load

# Load the trained model and vectorizer
model_file = '/content/Transposition.pkl'
vectorizer_file = '/content/Transposition_vectorizer.pkl'

try:
    model = load(model_file)
    vectorizer = load(vectorizer_file)
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading the model/vectorizer: {e}")
    exit()

# Load algorithm label categories
label_file = '/content/transposition_with_key.csv'  # Adjust path if needed
df = pd.read_csv(label_file)
algorithm_categories = pd.Categorical(df['Algorithm']).categories

# Predict the encryption algorithm
def predict_algorithm(plaintext, ciphertext, key):
    # Combine ciphertext and key into features
    features = ciphertext + ' ' + key

    # Transform features using the loaded vectorizer
    transformed_features = vectorizer.transform([features])

    # Predict the algorithm code
    predicted_code = model.predict(transformed_features)[0]

    # Map code to the actual algorithm name
    predicted_algorithm = algorithm_categories[predicted_code]
    
    return predicted_algorithm

# Interactive input for the use case
def main():
    print("=== Transposition Cipher Algorithm Prediction ===")
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
