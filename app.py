# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('rca_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_rca', methods=['POST'])
def predict_rca():
    df = pd.read_csv('telecom_logs.csv')
    df = df[df['status'] == 'ERROR']
    latest = df.tail(1)

    encoded = pd.get_dummies(latest[['component', 'error_type', 'severity']])
    for col in model.feature_names_in_:
        if col not in encoded:
            encoded[col] = 0
    encoded = encoded[model.feature_names_in_]

    extra_features = latest[['traffic_volume', 'packet_loss_rate', 'latency_ms']].values
    input_data = pd.concat([encoded, pd.DataFrame(extra_features, columns=['traffic_volume', 'packet_loss_rate', 'latency_ms'])], axis=1)

    prediction = model.predict(input_data)[0]
    return f"Predicted Root Cause: <b>{prediction}</b>"

@app.route('/reinforcement_learning', methods=['POST'])
def reinforcement_learning():
    import subprocess
    result = subprocess.run(['python', 'rl_update.py'], capture_output=True, text=True)
    return f"<pre>{result.stdout}</pre>"

@app.route('/log_parsing', methods=['POST'])
def log_parsing():
    return "Log Parsing module will be integrated by your teammate."

if __name__ == '__main__':
    app.run(debug=True)
