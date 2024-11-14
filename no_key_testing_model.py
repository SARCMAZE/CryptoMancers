import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load and prepare the testing dataset
file_path = r"/content/ciphertext_dataset_testing.csv"  # Path to your testing dataset
test_data = pd.read_csv(file_path)

# Ensure 'Ciphertext' column exists and clean the ciphertext format
test_data['Ciphertext'] = test_data['Ciphertext'].str.replace(' ', '')  # Remove spaces

# Encode target labels for consistency with training data
label_encoder = LabelEncoder()
test_data['Algorithm'] = label_encoder.fit_transform(test_data['Algorithm'])

# Separate features and target
X_test_raw = test_data['Ciphertext']
y_test = test_data['Algorithm']

# Transform ciphertext to numerical arrays
print("Converting ciphertext to numerical arrays for testing...")
X_test_transformed = []
for ciphertext in tqdm(X_test_raw, desc="Transforming Test Ciphertext"):
    byte_array = [int(ciphertext[i:i + 2], 16) for i in range(0, len(ciphertext), 2)]
    X_test_transformed.append(byte_array)

# Padding sequences to match training sequence length
max_len = 256  # Ensure this matches the length used during training
X_test_padded = pad_sequences(X_test_transformed, maxlen=max_len, padding='post', truncating='post')

# Evaluate the model
print("Evaluating model on test data...")
y_pred = model.predict(X_test_padded)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
