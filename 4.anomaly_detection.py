# anomaly_detection.py
import numpy as np
import pandas as pd
from model import create_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os

csv_path = "data/Test.csv"
weights_path = "models/global_model.weights.h5"
output_path = "results/anomaly_detection.csv"

expected_cols = [
    'day', 'breed', 'gender', 'farm'
]

df = pd.read_csv(csv_path)
df_orig = df.copy()

df = pd.get_dummies(df, columns=["breed", "gender", "farm"])
for col in expected_cols:
    if col not in df.columns:
        df[col] = 0
df = df[expected_cols]

print(f"Columnas alineadas: {df.columns.tolist()}")
print(f"Shape de entrada: {df.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
model = create_model(input_dim=len(expected_cols))
model.load_weights(weights_path)
preds = model.predict(X_scaled).flatten()
output_df = pd.DataFrame({
    "animal_id": df_orig.get("animal_id", pd.Series(range(len(preds)))),
    "day": df_orig["day"],
    "predicted_weight_kg": preds
})
if "weight" in df_orig.columns:
    output_df["real_weight_kg"] = df_orig["weight"]
    output_df["residual"] = output_df["real_weight_kg"] - output_df["predicted_weight_kg"]
    resid_std = output_df["residual"].std()
    threshold = 2 * resid_std
    output_df["anomaly"] = output_df["residual"].abs() > threshold
    num_anomalies = output_df["anomaly"].sum()
    mae = mean_absolute_error(output_df["real_weight_kg"], output_df["predicted_weight_kg"])
    print(f"MAE: {mae:.2f}")
    print(f"Anomalías detectadas: {num_anomalies} de {len(output_df)} registros")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"Resultados guardados en: {output_path}")
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(output_df["residual"], kde=True)
plt.axvline(x=threshold, color="red", linestyle="--", label="±2σ")
plt.axvline(x=-threshold, color="red", linestyle="--")
plt.title("Distribución de residuales")
plt.legend()
plt.show()