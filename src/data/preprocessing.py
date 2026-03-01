import pandas as pd

def preprocess_data(df):
    df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})
    
    X = df.drop('deposit', axis=1)
    y = df['deposit']
    
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y