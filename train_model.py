import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix




# Create models folder if not present
os.makedirs("models", exist_ok=True)

# 1 Load dataset
df = pd.read_csv("data/auto_insurance_claims.csv")
print("Dataset Loaded")
print("Columns:", df.columns)


# 2 Remove unwanted column
if "_c39" in df.columns:
    df = df.drop(columns=["_c39"])

# 3 Convert target variable
df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

# 4 Drop date columns (not useful for model)
drop_cols = ["policy_bind_date", "incident_date"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# 5 Encode categorical columns
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object" and col != "fraud_reported":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# 6 Define features and target
target_col = "fraud_reported"

X = df.drop(columns=[target_col])
y = df[target_col]

feature_cols = X.columns.tolist()

# 7 Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 8 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 9 Train model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 10 Evaluate
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report
print("\nClassification Report\n", classification_report(y_test, y_pred))

# 11 Save model
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("\nModel saved successfully!")

plt.figure(figsize=(8,6))
sns.barplot(x=model.feature_importances_, y=feature_cols)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.close()
print("Feature importance plot saved as 'models/feature_importance.png'")