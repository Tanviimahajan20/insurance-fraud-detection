from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# ---------------- LOAD MODEL ---------------- #
try:
    model = joblib.load("models/fraud_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")

    print("✅ Model loaded successfully")
    print("Features used by model:", feature_cols)

except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    feature_cols = []
    label_encoders = {}

# ---------------- ROUTES ---------------- #
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/choose_fraud')
def choose_fraud():
    return render_template('choose_fraud.html')

@app.route('/automobile')
def automobile():
    return render_template('automobile.html')

@app.route('/health')
def health():
    return render_template('health.html')

@app.route('/property')
def property_fraud():
    return render_template('property.html')

# ---------------- PREDICTION ---------------- #
@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()
        input_data = {}

        # Loop through all features expected by the model
        for col in feature_cols:
            # Get value from form or use a default if missing
            val = form_data.get(col)

            # Assign safe defaults if missing
            if val is None or val == "":
                if col in label_encoders:  # Categorical
                    val = list(label_encoders[col].classes_)[0]  # first class
                else:  # Numeric
                    val = 0

            # Encode categorical features
            if col in label_encoders:
                encoder = label_encoders[col]
                if str(val) in encoder.classes_:
                    val = encoder.transform([str(val)])[0]
                else:
                    val = 0  # fallback if unknown category

            # Convert numeric to float safely
            else:
                try:
                    val = float(val)
                except:
                    val = 0

            input_data[col] = val

        # Convert input_data to numpy array
        input_array = np.array(list(input_data.values())).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"

# ---------------- FEATURE IMPORTANCE ---------------- #
@app.route("/feature_importance")
def feature_importance():
    if model is None:
        return "Model not loaded. Cannot display feature importance."

    try:
        importances = model.feature_importances_
        plt.figure(figsize=(8,6))
        plt.barh(feature_cols, importances)
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()

        # Convert plot to base64 string for HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        plt.close()

        return f"<img src='data:image/png;base64,{encoded}'/>"
    except Exception as e:
        return f"Error generating feature importance: {str(e)}"

# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    app.run(debug=True)