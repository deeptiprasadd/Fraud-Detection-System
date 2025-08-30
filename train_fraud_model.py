# train_fraud_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("📂 Loading dataset...")
data = pd.read_csv("data/fraud_dataset.csv")

X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# Encode categorical
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train
print("🤖 Training model...")
model = LogisticRegression(max_iter=200, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Accuracy: {acc:.2f}")

# Save model + artifacts
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/fraud_scaler.pkl")
joblib.dump(list(X.columns), "models/fraud_features.pkl")
joblib.dump(acc, "models/fraud_accuracy.pkl")
joblib.dump(cm, "models/fraud_confusion.pkl")

print("✅ Model and metrics saved in /models/")
