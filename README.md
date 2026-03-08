# Insurance Fraud Detection

A **Flask web application** that predicts fraudulent insurance claims for **Automobile, Health, and Property** insurance.

---

## Features

- Predicts fraud using a **Random Forest Classifier**
- Handles **categorical and numerical inputs**
- Automatically manages **missing optional fields**
- Separate forms for **Automobile, Health, and Property** claims

---

## Technologies Used

- Python 3
- Flask
- Scikit-learn
- Pandas & NumPy
- SMOTE for class imbalance
- HTML/CSS for frontend

---

## Project Structure
insurance-fraud-detection/
│
├─ app.py # Flask application
├─ train_model.py # Model training script
├─ requirements.txt # Python dependencies
├─ templates/ # HTML templates
│ ├─ login.html
│ ├─ choose_fraud.html
│ ├─ automobile.html
│ ├─ health.html
│ └─ property.html
├─ static/ # CSS, images
│ └─ style.css
├─ data/ # Dataset folder
│ └─ auto_insurance_claims.csv
└─ models/ # Saved models
├─ fraud_model.pkl
├─ feature_cols.pkl
└─ label_encoders.pkl

