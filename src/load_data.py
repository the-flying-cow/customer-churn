import pandas as pd

def load(TRAIN_PATH):
    data = pd.read_csv(TRAIN_PATH)
    data.head()
    data.describe()
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    print(data.isnull().values.any())
    print(data.isnull().sum())
    print(data.duplicated().sum())

    data_copy = data.copy()
    data_copy= data_copy.drop(columns=["customerID"])

    return data_copy