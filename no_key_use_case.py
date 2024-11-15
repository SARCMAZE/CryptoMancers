import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the saved model and label encoder
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Function to convert ciphertext to numerical format
def transform_ciphertext(ciphertext):
    # Remove spaces (if any)
    ciphertext = ciphertext.replace(' ', '')
    # Convert hex string to byte array
    byte_array = [int(ciphertext[i:i + 2], 16) for i in range(0, len(ciphertext), 2)]
    return byte_array

# Function to calculate frequency transitions (custom feature)
def compute_frequency_transitions(ciphertext):
    transitions = sum(1 for i in range(1, len(ciphertext)) if ciphertext[i] != ciphertext[i - 1])
    return transitions

# Function to predict encryption algorithm
def predict_algorithm(ciphertext):
    # Transform the ciphertext
    transformed_cipher = transform_ciphertext(ciphertext)
    # Pad the sequence
    padded_cipher = pad_sequences([transformed_cipher], maxlen=128, padding='post', dtype='int32')
    # Compute frequency transitions
    frequency_transitions = compute_frequency_transitions(ciphertext)
    # Append the custom feature
    final_input = np.hstack([padded_cipher, np.array([[frequency_transitions]])])
    # Predict the algorithm
    prediction = rf_model.predict(final_input)
    # Decode the label
    predicted_algorithm = label_encoder.inverse_transform(prediction)
    return predicted_algorithm[0]

# Main execution
if __name__ == "__main__":
    # Prompt the user for a ciphertext input
    user_ciphertext = input("Enter the ciphertext (hexadecimal string): ").strip()
    try:
        # Predict the encryption algorithm
        predicted_algorithm = predict_algorithm(user_ciphertext)
        print(f"Predicted Encryption Algorithm: {predicted_algorithm}")
    except Exception as e:
        print(f"Error processing the input ciphertext: {e}")
 
