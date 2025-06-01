# rl_update.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data from predictions CSV
df = pd.read_csv('rca_predictions.csv')

# Reward mechanism
df['reward'] = df.apply(lambda row: 1 if row['predicted'] == row['actual'] else -1, axis=1)
print("Average Reward:", df['reward'].mean())

# Optional retraining if accuracy drops
correct = df[df['reward'] == 1]
features = correct.drop(['predicted', 'actual', 'reward'], axis=1)
labels = correct['actual']

# Re-train model with correct predictions
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, labels)

# Save updated model
joblib.dump(model, 'rca_model.pkl')
print("Model updated using RL and saved.")
