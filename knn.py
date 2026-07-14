import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────
df = pd.read_csv("data/listings_spatial.csv")
df = df.dropna(subset=["price"])

feature_cols = ["latitude", "longitude", "district_group"]
X = df[feature_cols]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["latitude", "longitude"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["district_group"]),
])

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# Baseline-Modell
# ─────────────────────────────────────────────
baseline = Pipeline([
    ("preprocess", preprocess),
    ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance")),
])

baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)

print("Baseline MAE:", mean_absolute_error(y_test, y_pred))
print("Baseline R2:", r2_score(y_test, y_pred))

# Beispielwerte
sample = pd.DataFrame({"actual": y_test, "predicted": y_pred}).sample(10, random_state=42)
sample["error"] = (sample["actual"] - sample["predicted"]).abs()
print(sample)

# ─────────────────────────────────────────────
# GridSearchCV – Hyperparameter Tuning
# ─────────────────────────────────────────────
pipe = Pipeline([
    ("preprocess", preprocess),
    ("knn", KNeighborsRegressor()),
])

param_grid = {
    "knn__n_neighbors": [1, 2, 3, 5, 7, 10, 15, 20, 30, 50],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

grid_search = GridSearchCV(
    pipe,
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print("Beste Parameter:", grid_search.best_params_)
print("Bester CV-MAE:", -grid_search.best_score_)

# Ergebnisse sortieren
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results["mae"] = -cv_results["mean_test_score"]
cv_results = cv_results[["param_knn__n_neighbors", "param_knn__weights", "param_knn__p", "mae"]]
cv_results.columns = ["k", "weights", "p", "mae"]
cv_results = cv_results.sort_values("mae").reset_index(drop=True)
print(cv_results.head())

# ─────────────────────────────────────────────
# Bestes Modell testen
# ─────────────────────────────────────────────
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_best, alpha=0.3, s=10)
lims = [0, max(y_test.max(), y_pred_best.max())]
ax.plot(lims, lims, color="red", linestyle="--", label="perfekte Vorhersage")
ax.set_xlabel("Tatsächlicher Preis")
ax.set_ylabel("Vorhergesagter Preis")
ax.set_title("Echte vs. vorhergesagte Preise (bestes Modell)")
ax.legend()
plt.show()
