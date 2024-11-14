import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras.preprocessing.sequence import pad_sequences
import pickle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the dataset
file_path = r"/content/ciphertext_dataset_training.csv"
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
    # Convert each hexadecimal value to integer, ensure it's limited by byte size
    byte_array = [int(ciphertext[i:i + 2], 16) for i in range(0, len(ciphertext), 2)]
    X_transformed.append(byte_array)

# Padding to
