import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load existing CSV
df = pd.read_csv('rl_dataset.csv')  # Should contain LogMessage and RootCause

# Retrain model
X = df['Component']
y = df['RootCause']

vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_vect, y)

# Save updated model
with open('rca_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("Reinforcement learning model updated and saved to rca_model.pkl")
