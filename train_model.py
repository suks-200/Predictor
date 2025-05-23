import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset (update with your CSV path)
df = pd.read_csv("Crop_recommendation.csv")

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the new model (replace old .pkl file)
with open("models/RandomForest.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved.")
