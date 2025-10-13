from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def model_predict(pipe, X_test):
    y_pred = pipe.predict(X_test)

    return y_pred

def hyper_parameter_tune(pipe, X_train, y_train ):
    params= {
        "classifier__max_depth": randint(20, 100),
        "classifier__max_features": ['sqrt', 'log2'],
        "classifier__max_leaf_nodes": randint(100, 500),
        "classifier__min_samples_leaf": randint(2, 20),
        "classifier__n_estimators": randint(50, 400),
        "classifier__min_samples_split": randint(2, 20),
        "classifier__criterion": ['gini', 'entropy']
        }

    random_search= RandomizedSearchCV(pipe, param_distributions=params, n_iter=100, scoring='roc_auc', cv=10, n_jobs=-1, random_state= 7)
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_

def predict_churn(model, test):
    preds= model.predict(test)
    return preds