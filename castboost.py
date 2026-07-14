from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# Loading Screen Funktion
# ─────────────────────────────────────────────
def loading(msg="Lade...", duration=4):
    for i in range(duration * 2):  # 0.5s Schritte
        sys.stdout.write(f"\r{msg} {'.' * (i % 4)}")
        sys.stdout.flush()
        time.sleep(0.5)
    print("\rFertig!            ")

# ─────────────────────────────────────────────
# 1. Daten laden
# ─────────────────────────────────────────────
df = pd.read_csv("data/combined_data.csv")

exclude = [
    "description", "price", "id",
    "booked_days", "booked_winter", "booked_summer",
    "booked_spring", "booked_fall",
    "booked_weekdays", "booked_weekenddays",
    "availability_365"
]

all_features = [col for col in df.columns if col not in exclude]

# ─────────────────────────────────────────────
# 2. Feature-Typen bestimmen
# ─────────────────────────────────────────────
categorical_features = []
text_features = []
numeric_features = []

for col in all_features:
    if df[col].dtype in ["int64", "float64"]:
        numeric_features.append(col)
    else:
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len < 30:
            categorical_features.append(col)
        else:
            text_features.append(col)

# Fehlende Werte bereinigen
for col in categorical_features:
    df[col] = df[col].fillna("Missing").astype(str).str.strip()

for col in text_features:
    df[col] = df[col].fillna("").astype(str).str.strip()

# ─────────────────────────────────────────────
# 3. Train/Test Split
# ─────────────────────────────────────────────
X_all = df[all_features]
y_all = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# Loading Screen vor dem Training
# ─────────────────────────────────────────────
loading("Starte CatBoost Grundmodell")

# ─────────────────────────────────────────────
# 4. CatBoost Grundmodell
# ─────────────────────────────────────────────
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    random_state=42,
    verbose=True
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    text_features=text_features,
    eval_set=(X_test, y_test)
)

# ─────────────────────────────────────────────
# 5. Evaluation Grundmodell
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# ─────────────────────────────────────────────
# 6. Hyperparameter Tuning – learning_rate
# ─────────────────────────────────────────────
loading("Hyperparameter Tuning: learning_rate")

learning_rates = [0.01, 0.05, 0.1, 0.2]
results_lr = []

for lr in learning_rates:
    print(f"\nTrainiere learning_rate={lr}")

    model_lr = CatBoostRegressor(
        iterations=300,
        learning_rate=lr,
        depth=8,
        loss_function="RMSE",
        random_state=42,
        verbose=True
    )

    model_lr.fit(
        X_train, y_train,
        cat_features=categorical_features,
        text_features=text_features,
        eval_set=(X_test, y_test)
    )

    y_pred_lr = model_lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    results_lr.append({"learning_rate": lr, "MAE": mae_lr, "RMSE": rmse_lr})

results_lr_df = pd.DataFrame(results_lr)
print("\nErgebnisse learning_rate:")
print(results_lr_df)

# ─────────────────────────────────────────────
# 7. Hyperparameter Tuning – depth
# ─────────────────────────────────────────────
loading("Hyperparameter Tuning: depth")

depths = [4, 6, 8, 10]
results_depth = []

for d in depths:
    print(f"\nTrainiere depth={d}")

    model_d = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=d,
        loss_function="RMSE",
        random_state=42,
        verbose=True
    )

    model_d.fit(
        X_train, y_train,
        cat_features=categorical_features,
        text_features=text_features,
        eval_set=(X_test, y_test)
    )

    y_pred_d = model_d.predict(X_test)
    mae_d = mean_absolute_error(y_test, y_pred_d)
    rmse_d = np.sqrt(mean_squared_error(y_test, y_pred_d))

    results_depth.append({"depth": d, "MAE": mae_d, "RMSE": rmse_d})

results_depth_df = pd.DataFrame(results_depth)
print("\nErgebnisse depth:")
print(results_depth_df)
