# Random Forest Customer Churn Project
The applciation uses a RandomForest model implemented on the Telco Customer Churn dataset from Kaggle.

# Description
The customer churn dataset contains information about customers, including various behavioral, service-related attributes along with labels indicating whether a customer discontinued their relation with the company or service. Here I have used a RandomForest model  for the classification purpose along with SMOTE to counter imbalanced dataset, of whether a customer will churn out or not.

## Setup
Create a virtual environment:
```bash
python -m venv .venv
```
## Run the project

In your terminal/command prompt, navigate to the root folder and execute the following 

```bash
pip install -r requirements.txt
```
then run, 

```bash
python -m src.Customer_Churn
```
If you are inside the src folder, then run

```bash
python Customer_Churn
```
