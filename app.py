from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import re
from flask_cors import CORS 

app = Flask(_name_)
CORS(app)

# Configure Gemini API key
genai.configure(api_key="AIzaSyDGrUHQSSeVt2YtD81SfAd9OjJkzOdg4-I")

# Load model and vectorizer
def load_model():
    with open("rca_model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# Simple log parser: extract timestamp, level, component, message (customize per your logs)
def parse_log(log):
    # Example regex for logs like: "2023-06-01 12:34:56 ERROR componentX: Error message details"
    pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})?\s*(?P<level>INFO|WARN|ERROR|DEBUG)?\s*(?P<component>\w+)?[:\-]?\s*(?P<message>.+)"
    match = re.match(pattern, log)
    if match:
        return match.groupdict()
    else:
        return {"raw_log": log}

@app.route("/")
def index():
    return render_template("index.html")

# 1. Log parser + insight generation endpoint
import google.generativeai as genai

genai.configure(api_key="AIzaSyDGrUHQSSeVt2YtD81SfAd9OjJkzOdg4-I")  # Make sure your API key is valid

# 1. Log parser + insight generation endpoint
@app.route("/parse_insight", methods=["POST"])
def parse_insight():
    log = request.form.get("log")
    if not log:
        return jsonify({"error": "No log provided"}), 400

    parsed = parse_log(log)

    prompt = f"""
    You are a helpful assistant. Given this parsed log data:
    {parsed}
    Provide detailed insights, possible root causes, and suggested actions.
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        insights = response.text  # or use response.candidates[0].text
    except Exception as e:
        insights = f"Insight generation failed: {str(e)}"

    return jsonify({
        "parsed_log": parsed,
        "insights": insights
    })

# 2. Root Cause Analysis (RCA)
@app.route("/rca", methods=["POST"])
def rca():
    component = request.form.get("log")
    if not component:
        return jsonify({"error": "No component/log provided"}), 400

    X_input = vectorizer.transform([component])
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input).max() * 100

    return jsonify({
        "prediction": pred,
        "confidence": round(proba, 2)
    })

# 3. Reinforcement Learning update with feedback
@app.route("/rl", methods=["POST"])
def rl():
    global vectorizer, model

    component = request.form.get("log")
    actual = request.form.get("actual")

    if not component or not actual:
        return jsonify({"error": "Component and actual root cause required"}), 400

    X_input = vectorizer.transform([component])
    pred = model.predict(X_input)[0]
    reward = 1 if pred == actual else -1

    new_entry = pd.DataFrame([{"Component": component, "RootCause": actual}])
    try:
        existing = pd.read_csv("rl_dataset.csv")
        updated = pd.concat([existing, new_entry], ignore_index=True)
    except FileNotFoundError:
        updated = new_entry
    updated.to_csv("rl_dataset.csv", index=False)

    # Retrain model on full dataset
    X_all = updated["Component"]
    y_all = updated["RootCause"]
    new_vect = TfidfVectorizer()
    X_vect_all = new_vect.fit_transform(X_all)

    new_model = RandomForestClassifier()
    new_model.fit(X_vect_all, y_all)

    with open("rca_model.pkl", "wb") as f:
        pickle.dump((new_vect, new_model), f)

    vectorizer, model = new_vect, new_model

    return jsonify({
        "prediction": pred,
        "reward": reward
    })

if __name__ == "__main__":
    app.run(debug=True)
