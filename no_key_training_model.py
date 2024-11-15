import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
file_path = r"/content/ciphertext_dataset_testing.csv"
data = pd.read_csv(file_path)

# Ensure 'Ciphertext' column exists and remove spaces in the ciphertext
data['Ciphertext'] = data['Ciphertext'].str.replace(' ', '')  # Remove spaces

# Label encoding for the algorithm column (target)
label_encoder = LabelEncoder()
data['Algorithm'] = label_encoder.fit_transform(data['Algorithm'])

# Split data into features (X) and target (y)
X = data['Ciphertext']  # Features
y = data['Algorithm']    # Target

# Feature transformation: Convert Ciphertext to numerical values
print("Converting ciphertext to numerical arrays...")
X_transformed = []
for ciphertext in tqdm(X, desc="Transforming Ciphertext"):
    byte_array = [int(ciphertext[i:i + 2], 16) for i in range(0, len(ciphertext), 2)]
    X_transformed.append(byte_array)

# Custom feature for Rail Fence: Calculate frequency transitions
def compute_frequency_transitions(ciphertext):
    transitions = sum(1 for i in range(1, len(ciphertext)) if ciphertext[i] != ciphertext[i - 1])
    return transitions

data['FrequencyTransitions'] = data['Ciphertext'].apply(lambda x: compute_frequency_transitions(x))

# Padding to make ciphertexts of equal length
print("Padding ciphertexts...")
X_padded = pad_sequences(X_transformed, maxlen=128, padding='post', dtype='int32')

# Append the custom feature
X_padded = np.hstack([X_padded, data['FrequencyTransitions'].values.reshape(-1, 1)])

# Split the data into train and test sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.15, random_state=42)

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, class_weight=class_weights_dict, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Model evaluation
print("Model evaluation results:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Calculate and display overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")
