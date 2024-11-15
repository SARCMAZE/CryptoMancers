import numpy as np
import pickle
from scipy.stats import entropy, skew, kurtosis
import pandas as pd
from joblib import load
from collections import Counter
import streamlit as st

# Load models and encoders
@st.cache_resource
def load_symmetric_model():
    try:
        model = pickle.load(open('symmetric.pkl', 'rb'))
        encoder = pickle.load(open('symmetric_label_encoder_k.pkl', 'rb'))
        return model, encoder
    except Exception as e:
        st.error(f"Error loading symmetric model or label encoder: {e}")
        return None, None

@st.cache_resource
def load_rf_model():
    try:
        return pickle.load(open('asymmetric.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading asymmetric model: {e}")
        return None

@st.cache_resource
def load_transposition_model():
    try:
        model = load('transposition.pkl')
        vectorizer = load('Transposition_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading Transposition model or vectorizer: {e}")
        return None, None

@st.cache_resource
def load_transposition_categories():
    try:
        df = pd.read_csv('transposition_with_key.csv')
        return pd.Categorical(df['Algorithm']).categories
    except Exception as e:
        st.error(f"Error loading transposition categories: {e}")
        return None

@st.cache_resource
def load_generic_model():
    try:
        model = pickle.load(open('mixed.pkl', 'rb'))
        encoder = pickle.load(open('generic_label_encoder.pkl', 'rb'))
        return model, encoder
    except Exception as e:
        st.error(f"Error loading generic model or label encoder: {e}")
        return None, None

# Load all resources
symmetric_model, label_encoder = load_symmetric_model()
rf_model = load_rf_model()
transposition_model, vectorizer = load_transposition_model()
algorithm_categories = load_transposition_categories()
generic_model, generic_label_encoder = load_generic_model()

# Helper functions
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

def predict_symmetric_algorithm(plaintext, ciphertext, key):
    features = np.array([extract_features(plaintext, ciphertext, key)])
    prediction = symmetric_model.predict(features)
    return label_encoder.inverse_transform(prediction)[0]

def transform_ciphertext(ciphertext):
    ciphertext = ciphertext.replace(' ', '')
    byte_array = [int(ciphertext[i:i + 2], 16) for i in range(0, len(ciphertext), 2)]
    return byte_array

def compute_frequency_transitions(ciphertext):
    transitions = sum(1 for i in range(1, len(ciphertext)) if ciphertext[i] != ciphertext[i - 1])
    return transitions

def predict_rf_algorithm(ciphertext):
    transformed_cipher = transform_ciphertext(ciphertext)
    frequency_transitions = compute_frequency_transitions(ciphertext)
    prediction = rf_model.predict([transformed_cipher + [frequency_transitions]])
    return label_encoder.inverse_transform(prediction)[0]

def predict_transposition_algorithm(plaintext, ciphertext, key):
    features = ciphertext + ' ' + key
    transformed_features = vectorizer.transform([features])
    predicted_code = transposition_model.predict(transformed_features)[0]
    return algorithm_categories[predicted_code]

def predict_generic_algorithm(ciphertext):
    features = np.array([transform_ciphertext(ciphertext)])
    prediction = generic_model.predict(features)
    return generic_label_encoder.inverse_transform(prediction)[0]

def check_frequency_match(plaintext, ciphertext):
    return Counter(plaintext) == Counter(ciphertext)

# Streamlit App
st.title("Encryption Algorithm Predictor")

# Input selection
st.sidebar.header("Select Prediction Mode")
prediction_mode = st.sidebar.selectbox("Choose input type:", ["With Plaintext", "Without Plaintext"])

if prediction_mode == "With Plaintext":
    st.subheader("Predict Algorithm With Plaintext")
    plaintext = st.text_area("Enter Plaintext:")
    ciphertext = st.text_area("Enter Ciphertext:")
    key = st.text_input("Enter Key:")
    key1 = st.text_input("Enter Key 1 (if asymmetric):")
    key2 = st.text_input("Enter Key 2 (if asymmetric):")

    if st.button("Predict"):
        if not plaintext or not ciphertext:
            st.error("Plaintext and Ciphertext are required.")
        elif key1 and key2:
            st.warning("Asymmetric prediction placeholder - define your model here.")
        elif key:
            if check_frequency_match(plaintext, ciphertext):
                result = predict_transposition_algorithm(plaintext, ciphertext, key)
                model_used = "Transposition"
            else:
                result = predict_symmetric_algorithm(plaintext, ciphertext, key)
                model_used = "Symmetric"
            st.success(f"Model Used: {model_used}")
            st.info(f"Prediction: {result}")
        else:
            st.error("A key is required.")

elif prediction_mode == "Without Plaintext":
    st.subheader("Predict Algorithm Without Plaintext")
    ciphertext = st.text_area("Enter Ciphertext:")

    if st.button("Predict"):
        if not ciphertext:
            st.error("Ciphertext is required.")
        else:
            try:
                result = predict_generic_algorithm(ciphertext)
                st.success(f"Model Used: Generic")
                st.info(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
