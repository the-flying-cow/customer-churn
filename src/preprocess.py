import pandas as pd

def preprocessing(data):
    data["TotalCharges"]= pd.to_numeric(data["TotalCharges"], errors= 'coerce')
    data["TotalCharges"]= data["TotalCharges"].fillna(0)
    
    return data