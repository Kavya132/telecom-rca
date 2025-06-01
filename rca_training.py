# rca_training.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('telecom_logs.csv')

# Filter error logs only
df = df[df['status'] == 'ERROR']

# Feature encoding
df_encoded = pd.get_dummies(df[['component', 'error_type', 'severity']])
features = pd.concat([df_encoded, df[['traffic_volume', 'packet_loss_rate', 'latency_ms']]], axis=1)
labels = df['root_cause']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'rca_model.pkl')

# Optional: save features for RL
X_test['predicted'] = model.predict(X_test)
X_test['actual'] = y_test.values
X_test.to_csv('rca_predictions.csv', index=False)

print("Model trained and saved as rca_model.pkl")
