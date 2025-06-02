import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('telecom_logs.csv')  # Make sure this has LogMessage, RootCause columns
X = df['Component']
y = df['RootCause']

# Vectorize
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_vect, y)

# Save model and vectorizer
with open('rca_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

# Save as CSV for reinforcement learning use
df.to_csv('rl_dataset.csv', index=False)

print("Model and vectorizer saved to rca_model.pkl and rl_dataset.csv")
