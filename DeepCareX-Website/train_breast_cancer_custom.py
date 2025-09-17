import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load your dataset
df = pd.read_csv("../DeepCareX-Datasets/Breast Cancer/subset_of_32_col.csv")

# Correct feature names from your CSV
features = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness"
]
target_col = "diagnosis"

X = df[features]
y = df[target_col]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Custom Breast Cancer Model trained successfully, accuracy: {accuracy:.2f}")

# Save model
out_dir = os.path.normpath(os.path.join("..", "Models", "Breast Cancer"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "breast_cancer.pkl")

joblib.dump(model, out_path)
print("📂 Model saved to:", out_path)

# Save feature order for Flask input
with open(os.path.join(out_dir, "features.txt"), "w") as f:
    f.write("\n".join(features))
print("📝 Saved feature names to features.txt")
