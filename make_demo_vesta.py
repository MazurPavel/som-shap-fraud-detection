import pandas as pd
import numpy as np
import pickle
import os

# Исходный Vesta-файл (тот, с которого ты обучал модель)
SOURCE_CSV = "data/vesta_sample_numeric.csv"

# Сколько строк хотим в демо
N_FRAUD = 30
N_NORMAL = 70

# --- загружаем scaler и список фич, сохранённые train_models.py ---
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# --- читаем исходные данные ---
df = pd.read_csv(SOURCE_CSV)

if "isFraud" not in df.columns:
    raise ValueError("В файле нет колонки isFraud – используй vesta_sample_numeric.csv")

fraud_df = df[df["isFraud"] == 1]
normal_df = df[df["isFraud"] == 0]

print("Всего fraud:", len(fraud_df), "normal:", len(normal_df))

n_fraud = min(N_FRAUD, len(fraud_df))
n_normal = min(N_NORMAL, len(normal_df))

fraud_sample = fraud_df.sample(n_fraud, random_state=42)
normal_sample = normal_df.sample(n_normal, random_state=42)

demo_df = pd.concat([fraud_sample, normal_sample])
demo_df = demo_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- скейлим фичи тем же scaler'ом ---
X_raw = demo_df[feature_cols].fillna(0.0).astype(float).values
X_scaled = scaler.transform(X_raw)

for i, col in enumerate(feature_cols):
    demo_df[col] = X_scaled[:, i]

os.makedirs("data", exist_ok=True)
demo_path = "data/demo_vesta.csv"
demo_df.to_csv(demo_path, index=False)

print("✅ Новый demo_vesta.csv сохранён:", demo_path)
print("Строк всего:", len(demo_df),
      "| fraud:", (demo_df["isFraud"] == 1).sum(),
      "| normal:", (demo_df["isFraud"] == 0).sum())