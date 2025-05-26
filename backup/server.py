# server.py
import flwr as fl
import numpy as np
import pandas as pd
from datetime import datetime

class RealisticStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=1,
            min_available_clients=1
        )

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None

        maes = []
        records = []

        for client_id, res in results:
            mae = res.metrics["mae"]
            client_name = res.metrics.get("client", f"client_{client_id}")
            maes.append(mae)
            records.append({
                "round": server_round,
                "client": client_name,
                "mae": mae,
                "timestamp": datetime.now().isoformat()
            })

        avg_mae = np.mean(maes)
        records.append({
            "round": server_round,
            "client": "Average",
            "mae": avg_mae,
            "timestamp": datetime.now().isoformat()
        })

        df = pd.DataFrame(records)
        df.to_csv(f"./results/results_round_{server_round}.csv", index=False)

        print(f"[Round {server_round}] MAE promedio: {avg_mae:.2f}")
        return 0.0, {"mae_avg": avg_mae}

# Iniciar servidor federado
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=RealisticStrategy()
)