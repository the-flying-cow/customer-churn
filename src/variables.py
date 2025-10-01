def variables(data_copy):
    X = data_copy.drop(columns = ["Churn"], axis=1)
    return X