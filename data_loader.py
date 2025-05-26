# data_loader.py
import pandas as pd

def load_farm_data(path):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=["breed", "gender", "farm"])
    X = df.drop(columns=["weight", "animal_id"])
    y = df["weight"]
    return X.to_numpy(), y.to_numpy()