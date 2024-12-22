from fastapi import FastAPI, Query
from enum import Enum
import pickle
import pandas as pd
# Initialize FastAPI app
app = FastAPI()

# Define Enums for Dropdown fields
class Sex(str, Enum):
    male = "male"
    female = "female"

class Housing(str, Enum):
    own = "own"
    rent = "rent"
    free = "free"

class SavingAccounts(str, Enum):
    little = "little"
    moderate = "moderate"
    quite_rich = "quite rich"
    rich = "rich"

class CheckingAccount(str, Enum):
    little = "little"
    moderate = "moderate"
    rich = "rich"

class Purpose(str, Enum):
    radio_tv = "radio/TV"
    education = "education"
    furniture_equipment = "furniture/equipment"
    new_car = "new car"
    used_car = "used car"
    business = "business"
    domestic_appliance = "domestic appliance"
    repairs = "repairs"
    vacation_others = "vacation/others"

# Load the trained model 
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Endpoint for predicting credit risk
@app.get("/predict")
async def predict_risk(
    age: int = Query(..., ge=18, le=100, description="Age of the individual (18-100)"),
    sex: Sex = Query(..., description="Sex of the individual ('male' or 'female')"),
    job: int = Query(..., ge=0, le=4, description="Job level (0-4)"),
    housing: Housing = Query(..., description="Housing status ('own', 'rent', 'free')"),
    saving_accounts: SavingAccounts = Query(..., description="Saving accounts ('little', 'moderate', 'quite rich', 'rich')"),
    checking_account: CheckingAccount = Query(..., description="Checking account status ('little', 'moderate', 'rich')"),
    credit_amount: int = Query(..., ge=0, description="Credit amount (>= 0)"),
    duration: int = Query(..., ge=0, description="Duration of credit (>= 0 months)"),
    purpose: Purpose = Query(..., description="Purpose of the credit ('radio/TV', 'education', etc.)")
):
    # Convert input data to a DataFrame
    input_data = {
        "age": [age],
        "sex": [0 if sex == "male" else 1],  # Encoding 'male' as 0, 'female' as 1
        "job": [job],
        "housing": [0 if housing == "own" else (1 if housing == "rent" else 2)],
        "saving_accounts": [saving_accounts.value],
        "checking_account": [checking_account.value],
        "credit_amount": [credit_amount],
        "duration": [duration],
        "purpose": [purpose.value]
    }
    df = pd.DataFrame(input_data)

    # Encode categorical fields
    df["saving_accounts"] = df["saving_accounts"].astype("category").cat.codes
    df["checking_account"] = df["checking_account"].astype("category").cat.codes
    df["purpose"] = df["purpose"].astype("category").cat.codes

    # Ensure data types match the model input
    df = df.astype(float)

    # Make prediction
    prediction = model.predict(df)
    risk = "Good Risk" if prediction[0] == 1 else "Bad Risk"

    return {
        "age": age,
        "sex": sex.value,
        "job": job,
        "housing": housing.value,
        "saving_accounts": saving_accounts.value,
        "checking_account": checking_account.value,
        "credit_amount": credit_amount,
        "duration": duration,
        "purpose": purpose.value,
        "prediction": risk
    }
