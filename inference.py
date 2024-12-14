import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
# Define a custom wrapper for XGBoost classifier
class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

# Step 1: Load the Input Data
input_file = '/home/aimssn-it/Desktop/Databeez/german_credit_data.csv'
data = pd.read_csv(input_file)

# Step 2: Data Preprocessing
# Handle missing values
data = data.assign(**{
    'Saving accounts': data['Saving accounts'].fillna('unknown'),
    'Checking account': data['Checking account'].fillna('unknown')
})

# Encode categorical variables
data_encoded = data.copy()
data_encoded = data_encoded.drop(['Unnamed: 0'], axis=1, errors='ignore')  # Drop if column exists
data_encoded['Sex'] = data_encoded['Sex'].map({'male': 0, 'female': 1})
data_encoded['Housing'] = data_encoded['Housing'].map({'own': 0, 'rent': 1, 'free': 2})
data_encoded['Saving accounts'] = data_encoded['Saving accounts'].astype('category').cat.codes
data_encoded['Checking account'] = data_encoded['Checking account'].astype('category').cat.codes
data_encoded['Purpose'] = data_encoded['Purpose'].astype('category').cat.codes
data_encoded['Risk'] = data_encoded['Risk'].map({'good': 1, 'bad': 0})  # Optional if target column exists

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

# Step 3: Load the Pretrained Model
model_path = '/home/aimssn-it/Desktop/Databeez/best_model.pkl' # Path to the saved model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Step 4: Perform Inference
# Drop target column if it exists during prediction
X = data_encoded.drop(['Risk'], axis=1, errors='ignore')  # 'Risk' is the target column
predictions = model.predict(X)

# Step 5: Save the Results
data=data.drop(['Unnamed: 0'], axis=1, errors='ignore')
results = data.copy()  # Retain original data for context
results['Prediction'] = predictions
output_file = 'model_results.csv'
results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
