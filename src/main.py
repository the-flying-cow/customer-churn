import os
from load_data import load
from eda_preprocess import eda_preprocess
from variables import variables
from model import model_predict, final_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR,"customerchurn_dataset", "Telco-Customer-Churn.csv")

def main():
    df= load(TRAIN_PATH)

    df= eda_preprocess(df)
    
    y = df["Churn"]
    data= variables(df)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state= 7, stratify= y)
    smote = SMOTE(random_state= 7)
    X_train_r, y_train_r = smote.fit_resample(X_train, y_train)
    
    preds= model_predict(X_train_r, y_train_r, X_test)
    
    final_report(y_test, preds)

if __name__=='__main__':
    main()
