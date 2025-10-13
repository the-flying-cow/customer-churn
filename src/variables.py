def variables(data):
    X = data.drop(columns = ["Churn"], axis=1)
    return X