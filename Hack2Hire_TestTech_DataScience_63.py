import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import dill
# Load the CSV file uploaded by the user
csv_file_path = '/home/aimssn-it/Desktop/Databeez/german_credit_data.csv'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Display the first few rows and basic information about the dataset
data_info = {
    "head": data.head(),
    "info": data.info(),
    "description": data.describe(include='all').T,
}

data_info

data=data.drop(['Unnamed: 0'],axis=1)

# Rename the columns to maintain consistency
data.rename(columns={'Age':'age',
                    'Sex':'sex',
                    'Job':'job',
                    'Housing':'housing',
                    'Saving accounts':'saving_accounts',
                    'Checking account':'checking_account',
                    'Credit amount':'credit_amount',
                    'Duration':'duration',
                    'Purpose':'purpose',
                    'Risk':'risk'
                    },
            inplace=True)

# numerical columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
num_cols

# categorical columns
cat_cols = data.select_dtypes(include='object').columns.tolist()
cat_cols

# Distribution of numerical columns

plt.figure(figsize=(16,10))
for colname in enumerate(num_cols):
    plt.subplot(2,2,colname[0]+1)
    sns.histplot(data[colname[1]], kde=True,color='blue')
    plt.title(colname[1])
    plt.tight_layout()
plt.tight_layout()
plt.title('Distribution of numerical columns')
plt.show()


# Make numerical variables normally distributed
data['age'] = np.sqrt(data['age'])
data['credit_amount'] = np.sqrt(data['credit_amount'])
data['duration'] = np.sqrt(data['duration'])
data['job'] = np.sqrt(data['job'])

plt.figure(figsize=(16,10))
for colname in enumerate(data[num_cols]):
    plt.subplot(2,2,colname[0]+1)
    sns.histplot(data[colname[1]], kde=True, color='blue')
    plt.title(colname[1])
    plt.tight_layout()
plt.tight_layout()
plt.title('Distribution of numerical columns')
plt.show()

# distribution of categorical columns
plt.figure(figsize=(16,10))
for colname in enumerate(cat_cols):
    plt.subplot(2,3,colname[0]+1)
    sns.countplot(x=data[colname[1]], data=data,hue='risk')
    plt.title(colname[1])
    plt.tight_layout()

# Piechart distribution of categorical columns

plt.figure(figsize=(30,30))
colors_account = ['red','blue','green','yellow','orange','purple','pink','brown']
for colname in enumerate(data.select_dtypes(exclude=[np.number]).columns):
    plt.subplot(3,3,colname[0]+1)
    # plt.style.use('dark_background')
    plt.pie(data[colname[1]].value_counts(), 
            labels=data[colname[1]].value_counts().index,
            colors=colors_account, 
            autopct='%1.1f%%', 
            pctdistance=0.85,
            wedgeprops=dict(width=0.99),
            textprops={'fontsize': 20})
       
    plt.legend(title=f'{colname[1]}',loc=2,fontsize='x-large')
    plt.tight_layout()  
    plt.title(colname[1])
plt.savefig('pie_chart.png')
# Histogram distribution of numerical columns
plt.figure(figsize=(30,30))
colors_account = ['red','blue','green','yellow','orange','purple','pink','brown']
plt.style.use('ggplot')
for colname in enumerate(data.select_dtypes(include=[np.number]).columns):
    plt.subplot(2,2,colname[0]+1)
    plt.hist(data[colname[1]].value_counts(), 
            bins=10,
            color=colors_account[colname[0]])  
    plt.xlabel(colname[1],fontsize=20),colors_account[colname[0]]
    plt.ylabel('Count',fontsize=20),colors_account[colname[0]]
    plt.legend(title=f'{colname[1]}',loc=1,fontsize='x-large')
    plt.title(colname[1])
plt.savefig('histogram.png')
plt.show()

# Check distribution of target variable 'Risk'
sns.countplot(x=data['risk'])
plt.show()

# Our target variable is imbalanced. We will handle this later 

# Handle missing values
data = data.assign(**{
    'saving_accounts': data['saving_accounts'].fillna('unknown'),
    'checking_account': data['checking_account'].fillna('unknown')
})

# Encode categorical variables
data_encoded = data.copy()
data_encoded['sex'] = data_encoded['sex'].map({'male': 0, 'female': 1})
data_encoded['housing'] = data_encoded['housing'].map({'own': 0, 'rent': 1, 'free': 2})
data_encoded['saving_accounts'] = data_encoded['saving_accounts'].astype('category').cat.codes
data_encoded['checking_account'] = data_encoded['checking_account'].astype('category').cat.codes
data_encoded['purpose'] = data_encoded['purpose'].astype('category').cat.codes
data_encoded['risk'] = data_encoded['risk'].map({'good': 1, 'bad': 0})

# Normalize or scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'job', 'credit_amount', 'duration']
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])


# Create correlation matrix
correlation_matrix = data_encoded.corr()

# Display the correlation matrix
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix,annot=True,
               cmap='coolwarm', 
               cbar=True, 
               annot_kws={'fontsize':15}, 
               vmin=-1, 
               vmax=1, 
               linewidths=.4,fmt='.3f')
plt.title('Correlation Matrix after encoding and scaling')
plt.show()

# Check correlation with the target variable 'Risk'
correlation_with_target = correlation_matrix['risk'].sort_values(ascending=False)


print("Correlation with target variable 'Risk':")
print(correlation_with_target)


# Separate the features and target variable
X = data_encoded.drop('risk', axis=1)
y = data_encoded['risk']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Update data_encoded with the resampled data
data_encoded = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['risk'])], axis=1)

# Verify the balance of the target variable
print(data_encoded['risk'].value_counts())

# Check distribution of target variable 'Risk' after handling class imbalance
sns.countplot(x=data_encoded['risk'])
plt.show()


# Split the dataset into training and testing sets
X = data_encoded.drop('risk', axis=1)
y = data_encoded['risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the XGBClassifier wrapper for sklearn compatibility
class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        # Define the parameters explicitly in the constructor
        self.model = XGBClassifier(  **kwargs)

        
    def fit(self, X, y):
        # Fit the XGBClassifier model
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # Predict using the fitted model
        return self.model.predict(X)

    def predict_proba(self, X):
        # Predict probabilities using the fitted model
        return self.model.predict_proba(X)

    def score(self, X, y):
        # Return the accuracy score using the model
        return self.model.score(X, y)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifierWrapper(eval_metric='logloss'),  # Using the wrapped XGBClassifier
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

# Train and evaluate models
best_model = None
best_score = 0
for name, model in models.items():
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
    
    if auc_roc > best_score:
        best_score = auc_roc
        best_model = model

# Save the best-performing model using dill

with open("best_model.pkl", "wb") as f:
    dill.dump(model, f)

print(f"Best model saved: {best_model}")
