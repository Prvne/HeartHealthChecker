import joblib
import numpy as np
import pandas as pd
import mysql.connector
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

# MySQL DB connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",  # <-- Replace with your MySQL password
    database="users"
)
cursor = db.cursor()

# Model paths
MODEL_PATH = "backend/models/heart_model.pkl"
LABEL_ENCODERS_PATH = "backend/models/label_encoders.pkl"
SCALER_PATH = "backend/models/scaler.pkl"

# Features
FEATURES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
CATEGORICAL_COLS = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
NUMERICAL_COLS = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Load model and preprocessors
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and preprocessing components loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or preprocessing components: {e}")
    model, label_encoders, scaler = None, None, None

@app.route("/")
def index():
    return redirect("/login")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        email = request.form["email"]

        cursor.execute("SELECT * FROM userdata WHERE username = %s", (username,))
        if cursor.fetchone():
            flash("Username already exists!", "warning")
            return redirect("/signup")

        cursor.execute("INSERT INTO userdata (username, password, email) VALUES (%s, %s, %s)",
                       (username, password, email))
        db.commit()
        flash("Account created! Please log in.", "success")
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password_input = request.form["password"]

        cursor.execute("SELECT password FROM userdata WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result and check_password_hash(result[0], password_input):
            session["user"] = username
            return redirect("/home")
        else:
            flash("Invalid username or password", "danger")
            return redirect("/login")
    return render_template("login2.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out", "info")
    return redirect("/login")

@app.route("/home")
def home():
    if "user" in session:
        username = session["user"]
        return render_template("home.html", username=username)
    else:
        return redirect("/login")

@app.route("/predict_page")
def predict_page():
    if "user" in session:
        username = session["user"]
        return render_template("heart10.html", username=username)
    else:
        return redirect("/login")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or label_encoders is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        for col in CATEGORICAL_COLS:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]

        input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result_message = (
            "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        )
        risk_level = get_risk_level(probability)
        probability = str(probability)
        probability = probability[:3]
        formatted_message = f"{result_message} (Risk Level: {risk_level}, Probability: {probability})"

        return jsonify({"redirect_url": url_for("results", result_message=formatted_message)})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


def get_risk_level(prob):
    if prob < 0.2:
        return "Low Risk"
    elif prob < 0.5:
        return "Moderate Risk"
    elif prob < 0.7:
        return "High Risk"
    else:
        return "Very High Risk"

@app.route("/about")
def about():
    return render_template("about.html", username=session.get("user", "Guest"))

@app.route("/faq")
def faq():
    return render_template("faq.html", username=session.get("user", "Guest"))

@app.route("/contact")
def contact():
    return render_template("contact.html", username=session.get("user", "Guest"))

@app.route("/professionals")
def professionals():
    return render_template("professionals.html", username=session.get("user", "Guest"))

@app.route("/results")
def results():
    result_message = request.args.get("result_message", "No result available.")
    return render_template("results.html", result_message=result_message)


if __name__ == "__main__":
    app.run(debug=True)
