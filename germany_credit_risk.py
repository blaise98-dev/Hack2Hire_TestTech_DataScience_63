
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import seaborn as sns
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load the dataset
csv_file_path = '/home/aimssn-it/Desktop/Databeez/german_credit_data.csv'
data = pd.read_csv(csv_file_path)

# Display dataset summary information
data_info = {
    "head": data.head(),  # First few rows of the dataset
    "info": data.info(),  # Info about columns (types, non-null counts)
    "description": data.describe(include='all')  # Summary statistics for numerical and categorical features
}

# Check data types and missing values
data_types = data.dtypes
missing_values = data.isnull().sum()
summary_statistics = data.describe(include='all')

# Store analysis results for later reference
data_analysis = {
    "data_types": data_types,
    "missing_values": missing_values,
    "summary_statistics": summary_statistics
}

# Data Preprocessing: Handle missing values and encoding categorical variables
data = data.assign(**{
    'Saving accounts': data['Saving accounts'].fillna('unknown'),
    'Checking account': data['Checking account'].fillna('unknown')
})

# Encode categorical variables and scale numerical ones
data_encoded = data.copy()
data_encoded = data_encoded.drop(['Unnamed: 0'], axis=1)
data_encoded['Sex'] = data_encoded['Sex'].map({'male': 0, 'female': 1})
data_encoded['Housing'] = data_encoded['Housing'].map({'own': 0, 'rent': 1, 'free': 2})
data_encoded['Saving accounts'] = data_encoded['Saving accounts'].astype('category').cat.codes
data_encoded['Checking account'] = data_encoded['Checking account'].astype('category').cat.codes
data_encoded['Purpose'] = data_encoded['Purpose'].astype('category').cat.codes
data_encoded['Risk'] = data_encoded['Risk'].map({'good': 1, 'bad': 0})

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

# Visualize correlations between features using a heatmap
correlation_matrix = data_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Display correlation with target variable 'Risk'
correlation_with_target = correlation_matrix['Risk'].sort_values(ascending=False)
print("Correlation with target variable 'Risk':")
print(correlation_with_target)

# Visualize pairwise relationships for selected features, with target variable hue
selected_features = ['Age', 'Job', 'Credit amount', 'Duration', 'Saving accounts', 'Purpose', 'Checking account', 'Risk']
sns.pairplot(data_encoded[selected_features], hue='Risk', palette='coolwarm')
plt.show()

# Split the data into features (X) and target (y)
X = data_encoded.drop('Risk', axis=1)
y = data_encoded['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Define a dictionary of models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifierWrapper(eval_metric='logloss'),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Extra Trees": ExtraTreesClassifier(),
}

# Model Evaluation: Train and evaluate models using cross-validation
best_model = None
best_score = 0
for name, model in models.items():
    # Perform cross-validation to evaluate models
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

    # Train and test the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("\n")

    # Track the best model based on AUC-ROC score
    if auc_roc > best_score:
        best_score = auc_roc
        best_model = model

# Save the best model to disk for future use
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print(f"Best model saved: {best_model}")
