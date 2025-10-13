import pandas as pd

def load(TRAIN_PATH):
    data = pd.read_csv(TRAIN_PATH)
    data= data.drop(columns=["customerID"])

    return data