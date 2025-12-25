import os
import pickle
import numpy as np
import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
from sklearn.metrics import f1_score

# ==============================
# НАСТРОЙКИ
# ==============================

# путь к обучающему датасету Vesta
TRAIN_CSV = "data/vesta_sample_numeric.csv"   # <-- твой Vesta-файл

SOM_X = 10
SOM_Y = 10
SOM_ITER = 5000        # можно уменьшить до 2000, если будет долго

# ==============================
# ЗАГРУЗКА ДАННЫХ
# ==============================

df = pd.read_csv(TRAIN_CSV)

# определяем колонку таргета
if "isFraud" in df.columns:
    target_col = "isFraud"
elif "Class" in df.columns:
    target_col = "Class"
else:
    raise ValueError("Не найдена колонка таргета isFraud или Class в CSV.")

# берём только числовые фичи
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c != target_col]

X_raw = df[feature_cols].fillna(0.0).astype(float)
y = df[target_col].astype(int)

print(f"Используем {len(feature_cols)} признаков, размер данных: {X_raw.shape}")

# ==============================
# СКЕЙЛИНГ
# ==============================

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw.values)

# ==============================
# ТРЕНИРУЕМ SOM
# ==============================

som = MiniSom(
    SOM_X,
    SOM_Y,
    X.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    neighborhood_function="gaussian",
    random_seed=42
)

som.random_weights_init(X)
print("Обучаем SOM ...")
som.train_random(X, SOM_ITER)
print("SOM готов.")

# ==============================
# ТРЕНИРУЕМ XGBoost
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist",      # нормально для Mac
    random_state=42
)

print("Обучаем XGBoost ...")
xgb_model.fit(X_train, y_train)
print("XGBoost готов.")

y_pred = xgb_model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"F1 на валидации: {f1:.3f}")

# ==============================
# SHAP EXPLAINER (через predict-функцию)
# ==============================

print("Строим SHAP explainer ...")

# небольшой background из обучающей выборки,
# чтобы SHAP работал быстрее и стабильнее
background = shap.sample(X_train, 100, random_state=42)

# ВАЖНО: объясняем функцию предсказаний модели,
# а не сам объект XGBClassifier — так обходим все баги совместимости.
explainer = shap.Explainer(xgb_model.predict, background)

# ==============================
# ГОТОВИМ DEMO DATAFRAME
# ==============================

# берём небольшую подвыборку для демо (например, 300 строк)
demo_indices = np.random.RandomState(42).choice(len(df), size=300, replace=False)

demo_df = df.iloc[demo_indices].copy()

# важно: кладём туда ИМЕННО скейленные фичи,
# чтобы потом в Streamlit можно было их сразу подавать в модель
X_demo = X[demo_indices]

for i, col in enumerate(feature_cols):
    demo_df[col] = X_demo[:, i]

demo_df.to_csv("data/demo_vesta.csv", index=False)
print("Demo CSV сохранён в data/demo_vesta.csv")

# ==============================
# СОХРАНЯЕМ МОДЕЛИ
# ==============================

os.makedirs("models", exist_ok=True)

with open("models/som_model.pkl", "wb") as f:
    pickle.dump(som, f)

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("models/shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("✅ Все модели и вспомогательные объекты сохранены в папку models/")