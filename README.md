# Loan_prediction

Loan Approval Prediction
This repository contains code for predicting the likelihood of loan approval based on applicant data using Logistic Regression. The goal is to build a binary classification model that predicts whether a loan will be approved (1) or denied (0).

Dataset
The dataset consists of two main files:

train.csv: Training dataset containing features and loan_status (target variable).
test.csv: Test dataset for which predictions need to be made.
Files
loan_prediction.ipynb: Jupyter notebook containing the Python code for data preprocessing, model training, evaluation, and prediction.
submission.csv: Sample submission file for predicting loan approval probabilities on the test dataset.
Requirements
Ensure you have Python 3.x installed along with the following libraries:

pandas
scikit-learn
numpy

You can install these dependencies using pip:
pip install pandas scikit-learn numpy
Usage
Clone the repository:

bash
Copy code
git clone copied_code
cd loan_prediction

Install dependencies:
Copy code
pip install -r requirements.txt
Run the notebook:
Open and run loan_prediction.ipynb in Jupyter notebook or any compatible environment (e.g., Google Colab).

Generate predictions:

Ensure train.csv and test.csv are placed in the same directory.
Run all cells in the notebook to preprocess data, train the Logistic Regression model, evaluate its performance, and generate predictions.
The predictions will be saved as submission.csv in the current directory.

WORKFLOW 

DATA PREPROCESSING :

Handle missing values.
Scale numeric features.
Encode categorical variables.

MODEL TRAINNING :

Initialize and train Logistic Regression model.
Use training data to fit the model.

MODEL EVALUATION :

Evaluate model performance using ROC AUC score, accuracy, and classification report on the validation set.

GENERATE PREDICTIONS :

Apply the trained model to predict probabilities on the test dataset.
Save predictions in submission.csv for submission.

The submission file submission.csv should contain predictions in the format:
python
id,loan_status
58645,0.5
58646,0.3
...

Ensure the submission.csv file is correctly formatted 


