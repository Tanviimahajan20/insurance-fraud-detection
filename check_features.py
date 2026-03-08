import joblib

cols = joblib.load("models/feature_cols.pkl")

print("Model features:")
print(cols)