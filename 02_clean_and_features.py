"""Step 2: Clean the data and create new features. Saves deliveries_clean.csv."""
import pandas as pd

df = pd.read_csv("deliveries.csv", parse_dates=["dispatch_time", "promised_time", "delivery_time"])

# 1. Remove duplicate records
df = df.drop_duplicates(subset="order_id")

# 2. Fill missing distances with the median of the same area
df["distance_km"] = df.groupby("area")["distance_km"].transform(lambda x: x.fillna(x.median()))

# 3. Missing rainfall -> assume 0 (no rain recorded)
df["rain_mm"] = df["rain_mm"].fillna(0)

# 4. New features
df["delivery_minutes"] = (df["delivery_time"] - df["dispatch_time"]).dt.total_seconds() / 60
df["hour"] = df["dispatch_time"].dt.hour
df["is_peak_hour"] = df["hour"].isin([8, 9, 17, 18, 19]).astype(int)
df["is_late"] = (df["delivery_time"] > df["promised_time"]).astype(int)

df.to_csv("deliveries_clean.csv", index=False)
print("Clean data saved:", df.shape)
print(df[["distance_km", "delivery_minutes", "is_late"]].describe())
