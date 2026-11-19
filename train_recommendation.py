
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/crops.csv")

print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())

# Expected columns
features = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]

target = "label"

# Validate dataset
missing = [c for c in features + [target] if c not in data.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Check your crops.csv column names."
    )

X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train
print("Training crop recommendation model...")

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/crop_recommendation_model.pkl"
)

print("\nCrop recommendation model saved successfully!")

