import streamlit as st
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

# Load the pretrained model
model_path = 'best_model.pkl'  # Path to the saved model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Germany Credit Risk Prediction system')

# User input
st.header('Enter the details to use application:')
age = st.number_input('Age', min_value=18, max_value=100, value=30)
job = st.number_input('Job', min_value=0, max_value=4, value=1)
credit_amount = st.number_input('Credit Amount', min_value=0, value=1000)
duration = st.number_input('Duration', min_value=0, value=12)
sex = st.selectbox('Sex', ['male', 'female'])
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving Accounts', ['little', 'moderate', 'quite rich', 'rich'])
checking_account = st.selectbox('Checking Account', ['little', 'moderate', 'rich'])
purpose = st.selectbox('Purpose', ['radio/TV', 'education', 'furniture/equipment', 'new car', 'used car', 'business', 'domestic appliance', 'repairs', 'vacation/others'])

# Encode categorical variables
sex = 0 if sex == 'male' else 1
housing = {'own': 0, 'rent': 1, 'free': 2}[housing]
saving_accounts = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3}[saving_accounts]
checking_account = {'little': 0, 'moderate': 1, 'rich': 2}[checking_account]
purpose = ['radio/TV', 'education', 'furniture/equipment', 'new car', 'used car', 'business', 'domestic appliance', 'repairs', 'vacation/others'].index(purpose)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Age': [age],
    'Job': [job],
    'Credit amount': [credit_amount],
    'Duration': [duration],
    'Sex': [sex],
    'Housing': [housing],
    'Saving accounts': [saving_accounts],
    'Checking account': [checking_account],
    'Purpose': [purpose]
})

# Ensure the feature order matches the training data
input_data = input_data[['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']]

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])

# Perform prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    st.write(f'Prediction: {"Good Credit" if prediction[0] == 1 else "Bad Credit"}')
    st.write(f'Prediction Probability: {prediction_proba[0]}')
