import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load Dataset
# Provide your dataset file path
input_file = 'D:/Quant Maze 2.0/transposition_with_key.csv'  # Update this to your dataset path
df = pd.read_csv(input_file)

# Step 2: Preprocess Data
# Combine 'Ciphertext' and 'Key' as features for prediction
df['Features'] = df['Ciphertext'] + ' ' + df['Key'].astype(str)

# Encode target variable (Algorithm)
df['Algorithm'] = df['Algorithm'].astype('category')
df['Algorithm_Code'] = df['Algorithm'].cat.codes

# Step 3: Feature Engineering (Text Vectorization)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))  # Character-level n-grams
X = vectorizer.fit_transform(df['Features'])
y = df['Algorithm_Code']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)

# Generate classification report as a dictionary
target_names = pd.Categorical(df['Algorithm']).categories
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

# Step 7: Display Precision for Each Algorithm
print("Precision for Each Algorithm:\n")
for algorithm in target_names:
    precision = report[algorithm]['precision']
    print(f"{algorithm}: {precision:.2f}")

# Step 8: Save the Model and Vectorizer (Optional)
# model and vectorizer can be saved for future use if needed
# from joblib import dump
# dump(model, 'crypto_algorithm_model.pkl')
# dump(vectorizer, 'crypto_vectorizer.pkl')
