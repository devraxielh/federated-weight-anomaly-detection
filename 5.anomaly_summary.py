import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("results/anomaly_detection.csv")
if "real_weight_kg" not in df.columns or "predicted_weight_kg" not in df.columns:
    raise ValueError("Faltan columnas de peso real o predicho.")

y_true = df["real_weight_kg"]
y_pred = df["predicted_weight_kg"]

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MAE : {mae:.2f} kg")
print(f"R²  : {r2:.4f}")

if "anomaly" in df.columns:
    anomalies = df["anomaly"].sum()
    total = len(df)
    percent_anomalies = (anomalies / total) * 100
    print(f"Anomalías detectadas: {anomalies} / {total} ({percent_anomalies:.2f}%)")

    df["abs_residual"] = (y_true - y_pred).abs()
    top_5 = df.sort_values("abs_residual", ascending=False).head(5)
    print("\nTop 5 animales más anómalos:")
    print(top_5[["animal_id", "day", "real_weight_kg", "predicted_weight_kg", "residual", "anomaly"]])
else:
    print("No se encontró la columna 'anomaly'. ¿Ya ejecutaste anomaly_detection.py?")