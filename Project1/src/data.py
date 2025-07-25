import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.dropna()  # axis = 1: remove cols
    # or df = df.fillna()

# Good for cleaner Pipelines - For Production pipelines, reusable modules
def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])       # columns.target - only for selecting 1 column
    y = df[target]                      # Optional: y = X.pop(target)
    return train_test_split(X, y, test_size, random_state=random_state)