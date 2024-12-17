# databeez_test_hack2hire

## Description
- This repository contains the code and resources for the `databeez_test_hack2hire` project.
- The dataset used is the German Credit Risk Dataset, a standard benchmark in credit risk modeling. It consists of 1,000 instances and 10 variables(9 features and 1 label).
-  This project leverages machine learning algorithms to analyze features such as  duration, credit amount,housing,checking account,saving account, sex, job, age, and purpose of credit.The system predicts whether a loan applicant is at "good risk" or "bad risk".
  
- Exploratory Data analysis report before preprocessing: [credit_risk_germany_data_profiling_report](./credit_risk_germany_data_profiling_report.html)
- Exploratory Data analysis report after preprocessing: [preprocessed_credit_risk_germany_data_profiling_report](./preprocessed_credit_risk_germany_data_profiling_report.html)
- The training was made on many classification algorithms and the best model is XGBOOST model with accuracy of 0.8~ 80%.
- Deployed application can be accessed on [hack2hiretesttechdatascience63](https://hack2hiretesttechdatascience63.streamlit.app/)


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
Follow these steps to install the project:

1. Clone the repository:
   ```sh
   git clone https://github.com/blaise98-dev/databeez_test_hack2hire.git

2. Navigate to the project directory:
   ```sh
   cd databeez_test_hack2hire

3. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
## Instructions on how to use the project:

1. Start the application:
2. Open your browser and navigate to the Jupyter Notebook URL provided in the terminal.
   ```sh
   jupyter notebook
   
## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```sh
   git checkout -b feature-branch

3. Commit your changes:
   ```sh
   git commit -m 'Add some feature'

4. Push to the branch:
   ```sh
    git push origin feature-branch

5. Open a pull request.

## License

This project is licensed under the MIT License.
 
