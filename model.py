import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 1. Daten laden und vorbereiten
df = pd.read_csv("data/listings_clean.csv")

# Freitext-Spalten rauswerfen (kann Modell noch nicht verarbeiten)
df = df.drop(["name", "description", "amenities"], axis=1)

# X (Features) und y (Zielvariable) trennen
X = df.drop("price", axis=1)
y = df["price"] 

print(X.shape)
print(y.head())

# 2. Train/Test-Split
# X_train, y_train -> 80% der Daten, zum Trainieren
# X_test, y_test -> 20% der Daten, zum Testen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training:  {X_train.shape}")
print(f"Test:      {X_test.shape}")

# 3. Random Forest trainieren
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # Modell lernt mit den Trainingsdaten

# 4. Evaluieren
y_pred = model.predict(X_test) # Macht Vorhersage für die Testdaten

mae = mean_absolute_error(y_test, y_pred) # Mean bsolute Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Root mean squared Error

print(f"MAE:  {mae:.2f}€")
print(f"RMSE: {rmse:.2f}€")

print(df["price"].describe()) # Verteilung anschauen

# Feature Importance
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance.head(10))
print(df["bathrooms"].describe())
print(df["bathrooms"].value_counts().head(10))
print(feature_importance.tail(10))

# Feature Importance Plot
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
feature_importance.plot(kind="barh")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Wichtigkeit")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("✅ Gespeichert als feature_importance.png")

