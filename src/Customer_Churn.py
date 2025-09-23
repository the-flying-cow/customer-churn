#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR,"customerchurn_dataset", "Telco-Customer-Churn.csv")


def main():
    data = pd.read_csv(TRAIN_PATH)

    data.head()
    data.describe()
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")


    print(data.isnull().values.any())
    print(data.isnull().sum())

    print(data.duplicated().sum())

    data_copy = data.copy()

    data_copy= data_copy.drop(columns=["customerID"])

    data_copy.head()

    sns.countplot(x="Churn",data= data_copy)
    plt.title("Churned Out")
    plt.show()

    obj_cols= []
    for col in data_copy.columns:
        if data[col].dtype=='object':
            obj_cols.append(col)
    num_cols= []
    for col in data_copy.columns:
        if data[col].dtype!='object':
            num_cols.append(col)
    print(f"Object columns: {obj_cols}\n")
    print(f"Numerical columns: {num_cols}")


    data_copy["TotalCharges"]= pd.to_numeric(data_copy["TotalCharges"], errors= 'coerce')
    data_copy["TotalCharges"]= data_copy["TotalCharges"].fillna(0)

    for col in obj_cols:
        print(f"Unique values in {col} are {data_copy[col].unique()}\n")

    plt.figure(figsize=(30, 20))
    for i,col in enumerate(num_cols,start= 1):
        plt.subplot(3,3,i)
        sns.boxplot(data= data_copy, hue= "Churn",y = col)
        plt.title(f"{col} vs 'Churn'")
    plt.show()

    plt.figure(figsize=(30, 20))
    for i,col in enumerate(num_cols,start= 1):
        plt.subplot(3,3,i)
        sns.histplot(data= data_copy, x= data_copy[col], bins= 15, color= 'yellow', edgecolor= 'blue')
        plt.ylabel("Frequency")
        plt.title(f"{col}'s Histogram")
    plt.show()

    plt.figure(figsize=(50,30))
    for i,col in enumerate(obj_cols, start=1):
        plt.subplot(5,5,i)
        sns.countplot(x=col,data= data_copy)
    plt.show()

    obj_dummy_cols=[]
    for col in obj_cols:
        if set(data_copy[col].unique()) <= {'Yes', 'No'} or set(data_copy[col].unique()) <= {'No', 'Yes'}:
            data_copy[col]= data_copy[col].astype(str).map({'Yes': 1, 'No': 0})
        else:
            obj_dummy_cols.append(col)

    data_copy["gender"]= data_copy["gender"].astype(str).map({'Female': 0, 'Male': 1}) 


    obj_dummy_cols.remove('gender')
    obj_dummy_cols.remove('TotalCharges')
    obj_dummy_cols


    for col in obj_dummy_cols:
        col_dummies = pd.get_dummies(data_copy[col],prefix= col,drop_first=True).astype(int)
        data_copy = pd.concat([data_copy,col_dummies], axis=1)

    data_copy = data_copy.drop(obj_dummy_cols,axis=1)


    corr_matrix = data_copy.corr()
    sns.clustermap(corr_matrix, annot=False, center=0, cmap="viridis",figsize= (30,30))
    plt.title("Correlation Clustermap")
    plt.show()


    y = data_copy["Churn"]
    X = data_copy.drop(columns = ["Churn"], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 7, stratify= y)
    smote = SMOTE(random_state= 7)
    X_train_r, y_train_r = smote.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators= 200, max_depth=None, random_state= 7, class_weight= "balanced")
    rf_model.fit(X_train_r, y_train_r)

    y_pred = rf_model.predict(X_test)

    print(f"Accuracy Score: {accuracy_score(y_pred, y_test)}")
    print(f"Classification Report:\n {classification_report(y_pred, y_test)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_pred, y_test)}")

if __name__=='__main__':
    main()