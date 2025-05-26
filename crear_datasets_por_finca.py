# crear_datasets_por_finca.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

data_dir = Path("./data")
files = list(data_dir.glob("Farm*.csv"))

for f in files:
    df = pd.read_csv(f)
    df = pd.get_dummies(df, columns=["breed", "gender", "farm"])
    X = df.drop(columns=["weight", "animal_id"])
    y = df["weight"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    np.save(f.with_suffix(".npy"), {"X": X_scaled, "y": y.to_numpy()})
    print(f"✅ Guardado: {f.with_suffix('.npy').name}")