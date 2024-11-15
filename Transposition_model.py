import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load

# Step 1: Load Dataset
input_file = 'D:/Quant Maze 2.0/transposition_with_key0.csv'  # Update this to your dataset path
df = pd.read_csv(input_file)

# Step 2: Preprocess Data
df['Features'] = df['Ciphertext'] + ' ' + df['Key'].astype(str)  # Combine 'Ciphertext' and 'Key'
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

# Step 6: Save the Model and Vectorizer
dump(model, 'crypto_algorithm_model.pkl')
dump(vectorizer, 'crypto_vectorizer.pkl')

# --- Testing the Model ---
# Step 7: Load the Saved Model and Vectorizer for Testing
try:
    model = load('crypto_algorithm_model.pkl')
    vectorizer = load('crypto_vectorizer.pkl')
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading the model/vectorizer: {e}")
    exit()

# Step 8: Load Test Data (Make sure your test data is preprocessed the same way)
test_file = 'D:/Quant Maze 2.0/transposition_with_keys.csv'  # Correct the path
 # Just the file name if in the same folder
  # Update this to your test dataset path
df_test = pd.read_csv(test_file)

# Step 9: Preprocess Test Data
df_test['Features'] = df_test['Ciphertext'] + ' ' + df_test['Key'].astype(str)  # Combine 'Ciphertext' and 'Key'
X_test_transformed = vectorizer.transform(df_test['Features'])  # Vectorize using the same vectorizer
y_test = df_test['Algorithm_Code']  # Assuming the 'Algorithm_Code' is available in the test data

# Step 10: Make Predictions
y_pred = model.predict(X_test_transformed)

# Step 11: Evaluate Model
target_names = pd.Categorical(df['Algorithm']).categories
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

# Step 12: Display Precision for Each Algorithm
print("Precision for Each Algorithm:\n")
for algorithm in target_names:
    precision = report[algorithm]['precision']
    print(f"{algorithm}: {precision:.2f}")
