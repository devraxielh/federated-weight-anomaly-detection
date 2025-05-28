# server.py
import flwr as fl
import numpy as np
from model import create_model
from datetime import datetime
from flwr.common.parameter import parameters_to_ndarrays

class FederatedNNStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=1,
            min_available_clients=1
        )

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters)

            input_shape = aggregated_weights[0].shape[0]
            model = create_model(input_dim=input_shape)
            model.set_weights(aggregated_weights)
            model.save_weights("./models/global_model.weights.h5")
            print(f"💾 Pesos del modelo global guardados en 'global_model_weights.h5'")
            return aggregated_parameters, {}
        else:
            return None, {}

    def aggregate_evaluate(self, server_round, results, failures):
        maes = [res.metrics["mae"] for _, res in results if res.metrics["mae"] is not None]
        if maes:
            avg_mae = np.mean(maes)
            print(f"[Round {server_round}] MAE promedio global: {avg_mae:.2f}")
            return 0.0, {"mae_avg": avg_mae}
        else:
            print(f"[Round {server_round}] No se pudo calcular MAE global")
            return 0.0, {}

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=FederatedNNStrategy()
)