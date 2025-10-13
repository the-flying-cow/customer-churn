from main import main
from model import predict_churn
import gradio as gr
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
if os.path.exists(MODEL_PATH):
    pipe = joblib.load(MODEL_PATH)
else:
    pipe = main()

def predict_inputs(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    
    if isinstance(SeniorCitizen, str):
        SeniorCitizen_num = 1 if SeniorCitizen.lower().startswith('y') else 0
    else:
        SeniorCitizen_num = int(SeniorCitizen)

    try:
        tenure = float(tenure)
    except Exception:
        tenure = 0.0
    try:
        MonthlyCharges = float(MonthlyCharges)
    except Exception:
        MonthlyCharges = 0.0
    try:
        TotalCharges = float(TotalCharges)
    except Exception:
        TotalCharges = 0.0

    test= pd.DataFrame([[gender, SeniorCitizen_num, Partner, Dependents, tenure, PhoneService, MultipleLines,
                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]], columns= features)

    pred = predict_churn(pipe, test)[0]
    return 'Churn' if int(pred) == 1 else 'No churn'

inputs = [
    gr.Radio(label='Gender', choices=['Male', 'Female']),
    gr.Radio(label='Senior Citizen', choices=['Yes', 'No']),
    gr.Radio(label='Partner', choices=['Yes', 'No']),
    gr.Radio(label='Dependents', choices=['Yes', 'No']),
    gr.Number(label='Tenure'),
    gr.Radio(label='Phone Service', choices=['Yes', 'No']),
    gr.Radio(label='Multiple Lines', choices=['Yes', 'No', 'No phone service']),
    gr.Radio(label='Internet Service', choices=['DSL', 'Fiber optic', 'No']),
    gr.Radio(label='Online Security', choices=['No', 'Yes', 'No internet service']),
    gr.Radio(label='Online Backup', choices=['Yes', 'No', 'No internet service']),
    gr.Radio(label='Device Protection', choices=['No', 'Yes', 'No internet service']),
    gr.Radio(label='Tech Support', choices=['No', 'Yes', 'No internet service']),
    gr.Radio(label='Streaming TV', choices=['No', 'Yes', 'No internet service']),
    gr.Radio(label='Streaming Movies', choices=['No', 'Yes', 'No internet service']),
    gr.Radio(label='Contract', choices=['Month-to-month', 'One year', 'Two year']),
    gr.Radio(label='Paperless Billing', choices=['Yes', 'No']),
    gr.Radio(label='Payment Method', choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
    gr.Number(label='Monthly Charges'),
    gr.Number(label='Total Charges')
]

gr.Interface(fn= predict_inputs, inputs= inputs, outputs= "text", 
            title= 'Telco Customer Churn',
            description='Current accuracy is only around 79%.. will be improved soon. Stay Tuned😉').launch(share=True)
