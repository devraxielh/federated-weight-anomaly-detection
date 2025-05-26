# client.py
import flwr as fl
from flwr.client import start_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_loader import load_farm_data
from sklearn.exceptions import NotFittedError
import numpy as np
import sys
import os
import joblib

class RFClient(fl.client.NumPyClient):
    def __init__(self, path):
        self.path = path
        self.farm_name = os.path.splitext(os.path.basename(path))[0]
        self.X, self.y = load_farm_data(path)
        self.model = RandomForestRegressor(n_estimators=50)

    def get_parameters(self, config): return []

    def fit(self, parameters, config):
        self.model.fit(self.X, self.y)
        joblib.dump(self.model, f"./models/{self.farm_name.lower()}_model.pkl")
        print(f"✅ Modelo guardado: {self.farm_name.lower()}_model.pkl")
        return [], len(self.X), {}

    def evaluate(self, parameters, config):
        try:
            preds = self.model.predict(self.X)
            mae = mean_absolute_error(self.y, preds)
            print(f"[{self.farm_name}] MAE local: {mae:.2f}")
            return 0.0, len(self.X), {"mae": mae, "client": self.farm_name}
        except NotFittedError:
            print(f"[{self.farm_name}] ⚠️ Modelo no entrenado aún. Se omite evaluación inicial.")
            return 0.0, len(self.X), {"mae": None, "client": self.farm_name}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Debes pasar el archivo CSV de la finca. Ejemplo: python client.py Farm1.csv")
        sys.exit(1)

    path = sys.argv[1]
    client = RFClient(path)
    start_client(server_address="localhost:8080", client=client.to_client())