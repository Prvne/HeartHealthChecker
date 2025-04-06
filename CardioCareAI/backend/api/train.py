from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle

def gmm_oversample(X, y, minority_class=1, n_samples=100):
    X_minority = X[y == minority_class]

    # Fit GMM to the minority class
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X_minority)

    # Sample new synthetic data points
    synthetic_samples, _ = gmm.sample(n_samples)

    # Combine synthetic samples with original data
    X_augmented = np.vstack((X, synthetic_samples))
    y_augmented = np.hstack((y, [minority_class] * n_samples))

    return shuffle(X_augmented, y_augmented, random_state=42)


# Load your dataset
df = pd.read_csv('backend/data/heart.csv')  # Replace with your dataset

# Features and target
FEATURES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
TARGET = 'HeartDisease'

X = df[FEATURES]
y = df[TARGET]

# Encode categorical features
label_encoders = {}
for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X_scaled[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']]
)

# Oversample with GMM
X_oversampled, y_oversampled = gmm_oversample(X_scaled.values, y.values, minority_class=1, n_samples=200)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessors
joblib.dump(model, 'backend/models/heart_model.pkl')
joblib.dump(label_encoders, 'backend/models/label_encoders.pkl')
joblib.dump(scaler, 'backend/models/scaler.pkl')


print(f"✅ Model trained and saved successfully!")
print(f"✅ Label Encoders saved to backend/models/label_encoders.pkl")
print(f"✅ Scaler saved to backend/models/scaler.pkl")


print(f"✅ Model trained and saved successfully!")
print(f"✅ Label Encoders saved to backend/models/label_encoders.pkl")
print(f"✅ Scaler saved to backend/models/scaler.pkl")
