"""Step 5: Predict delivery time with a Random Forest regression model."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("deliveries_clean.csv")

features = ["distance_km", "hour", "is_peak_hour", "rain_mm", "parcels_on_route"]
X, y = df[features], df["delivery_minutes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE (minutes):", round(mean_absolute_error(y_test, pred), 2))
print("R-squared    :", round(r2_score(y_test, pred), 3))
print("\nFeature importance:")
for name, imp in sorted(zip(features, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:18s} {imp:.3f}")
