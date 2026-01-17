import os
from load_data import load
from preprocess import preprocessing
from variables import variables
from model import hyper_parameter_tune
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR,"customerchurn_dataset", "Telco-Customer-Churn.csv")

def main():
    df= load(TRAIN_PATH)

    obj_cols= []
    for col in df.columns:
        if df[col].dtype=='object':
            obj_cols.append(col)

    num_cols= []
    for col in df.columns:
        if df[col].dtype!='object':
            num_cols.append(col)

    df= preprocessing(df)
    obj_cols.remove('TotalCharges')
    obj_cols.remove('Churn')
    num_cols.append('TotalCharges')
    
    
    preprocessor= ColumnTransformer([ ("num", StandardScaler(), num_cols), ("categorical", OneHotEncoder(), obj_cols)])

    df["Churn"]= df["Churn"].map({"No": 0, "Yes": 1})
    df["Churn"]= df["Churn"].fillna(0).astype(int)


    y = df["Churn"]
    data= variables(df)
    
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state= 7, stratify= y)

    rf_pipe= Pipeline([("preprocessor", preprocessor), ("smote_resample", SMOTE(random_state= 7)), ("classifier", RandomForestClassifier(class_weight='balanced', random_state= 7))])
    rf_pipe= hyper_parameter_tune(rf_pipe, X_train, y_train)

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(rf_pipe, model_path)    

    return rf_pipe

if __name__=='__main__':
    main()
