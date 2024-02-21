import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def train_test_split(df,cutoff_dt):
    
    X_train = df[df['pickup_hour'] < cutoff_dt].iloc[:,:-1]
    X_train.reset_index(drop=True,inplace=True)

    y_train = df[df['pickup_hour'] < cutoff_dt].iloc[:,-1]
    y_train.reset_index(drop=True,inplace=True)

    X_test = df[df['pickup_hour'] >= cutoff_dt].iloc[:,:-1]
    X_test.reset_index(drop=True,inplace=True)

    y_test = df[df['pickup_hour'] >= cutoff_dt].iloc[:,-1]
    y_test.reset_index(drop=True,inplace=True)
    
    return X_train,y_train,X_test,y_test
