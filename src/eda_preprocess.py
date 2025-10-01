import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

def eda_preprocess(data_copy):
    sns.countplot(x="Churn",data= data_copy)
    plt.title("Churned Out")
    plt.show()

    obj_cols= []
    for col in data_copy.columns:
        if data_copy[col].dtype=='object':
            obj_cols.append(col)
    num_cols= []
    for col in data_copy.columns:
        if data_copy[col].dtype!='object':
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

    return data_copy