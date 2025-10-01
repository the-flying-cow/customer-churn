from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def model_predict(X_train_r, y_train_r, X_test):
    rf_model = RandomForestClassifier(n_estimators= 200, max_depth=None, random_state= 7, class_weight= "balanced")
    rf_model.fit(X_train_r, y_train_r)
    y_pred = rf_model.predict(X_test)

    return y_pred

def final_report(y_pred, y_test):
    print(f"Accuracy Score: {accuracy_score(y_pred, y_test)}")
    print(f"Classification Report:\n {classification_report(y_pred, y_test)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_pred, y_test)}")
