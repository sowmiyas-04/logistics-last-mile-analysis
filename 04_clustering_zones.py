"""Step 4: Group delivery addresses into zones using K-Means clustering."""
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("deliveries_clean.csv")

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df["zone"] = kmeans.fit_predict(df[["latitude", "longitude"]])

summary = df.groupby("zone").agg(
    parcels=("order_id", "count"),
    avg_km=("distance_km", "mean"),
    late_rate=("is_late", "mean"),
).round(2)
print(summary)
