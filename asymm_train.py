import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import entropy
import os

# Load dataset (update file path as needed)
data_path = r"/content/asymmetric_encryption_dataset_with_keys_mediocre.csv"
data = pd.read_csv(data_path)

# Ensure the dataset loads correctly
if data.empty:
    raise ValueError("Dataset is empty. Please check the file path and contents.")

# Handle missing values in keys
data['Public Key'] = data['Public Key'].fillna("unknown")
data['Private Key'] = data['Private Key'].fillna("unknown")

# Define entropy function for randomness in ciphertext
def calculate_entropy(text):
    if not text:  # Handle empty strings
        return 0
    probability_distribution = np.bincount(bytearray(text.encode('utf-8'))) / len(text)
    return entropy(probability_distribution, base=2)

# Feature Engineering
data['Plaintext_Length'] = data['Plaintext'].apply(len)
data['Ciphertext_Length'] = data['Ciphertext'].apply(len)
data['Ciphertext_Entropy'] = data['Ciphertext'].apply(calculate_entropy)

# Vectorize 'Public Key' and 'Private Key' using TF-IDF
tfidf_vectorizer_pub = TfidfVectorizer(max_features=50)
tfidf_vectorizer_pri = TfidfVectorizer(max_features=50)

public_key_tfidf = tfidf_vectorizer_pub.fit_transform(data['Public Key']).toarray()
private_key_tfidf = tfidf_vectorizer_pri.fit_transform(data['Private Key']).toarray()

# Save the TF-IDF vectorizers for reuse
with open("tfidf_vectorizer_pub.pkl", "wb") as pub_file:
    pickle.dump(tfidf_vectorizer_pub, pub_file)

with open("tfidf_vectorizer_pri.pkl", "wb") as pri_file:
    pickle.dump(tfidf_vectorizer_pri, pri_file)

print("TF-IDF vectorizers saved successfully!")

# Prepare the features matrix X
X = pd.DataFrame({
    'Plaintext_Length': data['Plaintext_Length'],
    'Ciphertext_Length': data['Ciphertext_Length'],
    'Ciphertext_Entropy': data['Ciphertext_Entropy']
})

# Concatenate the TF-IDF features
X = pd.concat([
    X,
    pd.DataFrame(public_key_tfidf, columns=[f"public_key_{i}" for i in range(public_key_tfidf.shape[1])]),
    pd.DataFrame(private_key_tfidf, columns=[f"private_key_{i}" for i in range(private_key_tfidf.shape[1])])
], axis=1)

# Normalize numerical features
scaler = StandardScaler()
X[['Plaintext_Length', 'Ciphertext_Length', 'Ciphertext_Entropy']] = scaler.fit_transform(
    X[['Plaintext_Length', 'Ciphertext_Length', 'Ciphertext_Entropy']]
)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Algorithm'])

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model training with Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model and label encoder
model_file_path = "asymmetric_key_model.pkl"
label_encoder_file_path = "label_encoder.pkl"

with open(model_file_path, "wb") as model_file:
    pickle.dump(rf_model, model_file)

with open(label_encoder_file_path, "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print(f"Model and label encoder saved to {os.getcwd()}.")

# Cross-Validation to check generalization
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=kfold, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Predictions and Evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Verify saved files
print(f"Files in directory: {os.listdir()}")
