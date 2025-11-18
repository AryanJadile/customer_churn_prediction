# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load Data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Clean Data
# 'TotalCharges' sometimes has blank strings. Force them to numbers.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 3. Select Key Features (We focus on these 5 for the app)
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']
target = 'Churn'

X = df[features]
y = df[target]

# 4. Preprocessing (Convert text to numbers)
# We need to save these encoders to use them in the app later!
encoders = {}
for col in ['Contract', 'PaymentMethod']:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode Target (Yes/No -> 1/0)
y = y.map({'Yes': 1, 'No': 0})

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save Model and Encoders
joblib.dump(model, 'churn_model.pkl')
joblib.dump(encoders, 'encoders.pkl')

print("✅ Model trained and saved as 'churn_model.pkl'")
print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")