# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define column names (as used in app.py)
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Load dataset (Pima Indians Diabetes Dataset)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, names=columns)

# Drop unused columns (matches input format in your app.py)
df = df.drop(["SkinThickness", "DiabetesPedigreeFunction"], axis=1)

# ✅ CRITICAL FIX: Match the exact column order used in app.py
# In app.py, you create: [[preg, glucose, bp, insulin, bmi, age]]
# So the column order should be: ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
feature_columns = ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
X = df[feature_columns]  # Select features in the correct order
y = df["Outcome"]

print("Dataset shape:", X.shape)
print("Feature columns:", list(X.columns))
print("Sample data:")
print(X.head())

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance)

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/trained_model.pkl")

print("\n✅ Model trained and saved to model/trained_model.pkl")

# Test prediction with sample data (same format as app.py)
print("\n🧪 Testing prediction with sample data:")
sample_data = pd.DataFrame([[2, 120, 80, 30, 25.0, 35]], 
                          columns=feature_columns)
sample_prediction = model.predict(sample_data)
sample_prob = model.predict_proba(sample_data)

print(f"Sample input: {sample_data.iloc[0].to_dict()}")
print(f"Prediction: {sample_prediction[0]} (0=No Diabetes, 1=Diabetes)")
print(f"Probability: {sample_prob[0]}")