import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# Load the saved model and label encoder
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the testing dataset
test_file_path = r"/content/ciphertext_dataset_testing.csv"  # Path to the test dataset
test_data = pd.read_csv(test_file_path)

# Ensure 'Ciphertext' column exists and remove spaces in the ciphertext
test_data['Ciphertext'] = test_data['Ciphertext'].str.replace(' ', '')

# Feature transformation: Convert Ciphertext to numerical arrays
print("Converting ciphertext to numerical arrays...")
X_test_transformed = []
for ciphertext in tqdm(test_data['Ciphertext'], desc="Transforming Ciphertext"):
    byte_array = [int(ciphertext[i:i + 2], 16) for i in range(0, len(ciphertext), 2)]
    X_test_transformed.append(byte_array)

# Custom feature for Rail Fence: Calculate frequency transitions
def compute_frequency_transitions(ciphertext):
    transitions = sum(1 for i in range(1, len(ciphertext)) if ciphertext[i] != ciphertext[i - 1])
    return transitions

test_data['FrequencyTransitions'] = test_data['Ciphertext'].apply(lambda x: compute_frequency_transitions(x))

# Padding to make ciphertexts of equal length
print("Padding ciphertexts...")
X_test_padded = pad_sequences(X_test_transformed, maxlen=128, padding='post', dtype='int32')

# Append the custom feature
X_test_padded = np.hstack([X_test_padded, test_data['FrequencyTransitions'].values.reshape(-1, 1)])

# Predict on the test set
y_pred = rf_model.predict(X_test_padded)

# Decode the predicted labels
predicted_labels = label_encoder.inverse_transform(y_pred)

# If the true labels are available, evaluate the model
if 'Algorithm' in test_data.columns:
    y_test = label_encoder.transform(test_data['Algorithm'])  # Encode true labels
    print("Model evaluation results:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.2f}")
else:
    print("Predictions completed. True labels not provided in the test dataset.")

# Save predictions to a CSV file
output_file = "test_predictions.csv"
test_data['Predicted_Algorithm'] = predicted_labels
test_data.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
