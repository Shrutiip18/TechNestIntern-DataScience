import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('heart.csv')
df.columns = df.columns.str.strip()  

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy:.2f}")

# Save model and scaler
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('feature_name.pkl','wb') as f:
    pickle.dump(X.columns.tolist(), f)

print(" Model and scaler saved successfully.")
