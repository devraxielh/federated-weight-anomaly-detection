# client_nn.py
import flwr as fl
import numpy as np
from model import create_model
from sklearn.preprocessing import StandardScaler
import sys
import os

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, farm_path):
        self.farm_name = os.path.splitext(os.path.basename(farm_path))[0]
        data = np.load(farm_path, allow_pickle=True).item()
        self.X_train, self.y_train = data["X"], data["y"]
        self.model = create_model(self.X_train.shape[1])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        print(f"[{self.farm_name}] MAE local: {mae:.2f}")
        return loss, len(self.X_train), {"mae": mae, "client": self.farm_name}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Debes pasar el archivo .npy de la finca. Ejemplo: python client.py Farm1.npy")
        sys.exit(1)

    farm_path = sys.argv[1]
    fl.client.start_client(server_address="localhost:8080", client=FederatedClient(farm_path).to_client())