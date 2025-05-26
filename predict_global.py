# predict_global.py
import numpy as np
import pandas as pd
from model import create_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os

csv_path = "data/Test.csv"
weights_path = "models/global_model.weights.h5"
output_path = "results/global_predictions.csv"

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

# Verificación
print(f"✔️ Columnas alineadas: {list(df.columns)}")
print(f"✔️ Shape del input: {df.shape}")

# --- Escalado ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- Cargar modelo y pesos ---
model = create_model(input_dim=len(expected_cols))
model.load_weights(weights_path)

# --- Predicción ---
preds = model.predict(X_scaled).flatten()

# --- Construir salida ---
output_df = pd.DataFrame({
    "day": df_orig["day"],
    "predicted_weight_kg": preds
})

# --- Comparar con peso real si está disponible ---
if "weight" in df_orig.columns:
    output_df["real_weight_kg"] = df_orig["weight"]
    mae = mean_absolute_error(output_df["real_weight_kg"], output_df["predicted_weight_kg"])
    print(f"📏 MAE del modelo global: {mae:.2f}")

# --- Guardar resultado ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"✅ Predicciones guardadas en: {output_path}")