# Random Forest Customer Churn Project
The applciation uses a RandomForest model implemented on the Telco Customer Churn dataset from Kaggle.

# Description
The customer churn dataset contains information about customers, including various behavioral, service-related attributes along with labels indicating whether a customer discontinued their relation with the company or service. Here I have used a RandomForest model  for the classification purpose along with SMOTE to counter imbalanced dataset, of whether a customer will churn out or not.

## Directory Structure
```bash
src/
│── load_data.py        # functions to load Customer Churn dataset
│── eda_preprocess.py   # eda visualizations & preprocessing steps
│── variables.py        # feature definitions
│── model.py            # model training, prediction, and reporting
│── main.py             # orchestrates the pipeline
customerchurn_dataset/  # dataset folder
README.md               # documentation
requirements.txt        # specifies the dependencies
```

## Setup

```bash
git clone https://github.com/the-flying-cow/customer-churn.git
cd customer-churn
```

Create a virtual environment:
```bash
python -m venv .venv
venv\Scripts\activate
```
## Run the project

In your terminal/command prompt, navigate to the root folder and execute the following 

```bash
pip install -r requirements.txt
```

Inside the src folder, run the following

```bash
python main.py
```
