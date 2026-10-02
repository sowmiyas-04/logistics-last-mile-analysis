"""Generate a simulated last-mile delivery dataset for QuickCart Chennai.

Real company data is confidential, so this script creates a realistic
hypothetical dataset (3,000 deliveries) with a few deliberate data-quality
problems (missing values, duplicates) to be fixed in the cleaning step.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

areas = {  # area: (latitude, longitude)
    "Avadi": (13.1067, 80.0970), "Ambattur": (13.1143, 80.1548),
    "Tambaram": (12.9249, 80.1000), "Velachery": (12.9815, 80.2180),
    "Anna Nagar": (13.0850, 80.2101), "Adyar": (13.0012, 80.2565),
    "Porur": (13.0382, 80.1565), "Perungudi": (12.9654, 80.2461),
}
names = list(areas)
area = rng.choice(names, N)
lat = np.array([areas[a][0] for a in area]) + rng.normal(0, 0.015, N)
lon = np.array([areas[a][1] for a in area]) + rng.normal(0, 0.015, N)

dispatch = pd.Timestamp("2026-10-01 08:00") + pd.to_timedelta(
    rng.integers(0, 12 * 60, N), unit="m") + pd.to_timedelta(rng.integers(0, 28, N), unit="D")
distance = np.clip(rng.gamma(3, 3, N), 0.5, 40)
hour = dispatch.hour
peak = np.isin(hour, [8, 9, 17, 18, 19]).astype(int)
rain = np.where(rng.random(N) < 0.2, rng.gamma(2, 6, N), 0)
parcels = rng.integers(5, 40, N)

minutes = 12 + 3.2 * distance + 10 * peak + 0.5 * rain + 0.6 * parcels + rng.normal(0, 6, N)
minutes = np.clip(minutes, 10, None)
delivery = dispatch + pd.to_timedelta(minutes, unit="m")
promised = dispatch + pd.to_timedelta(60 + rng.integers(-10, 30, N), unit="m")

attempts = np.where(rng.random(N) < 0.12 + 0.05 * peak, 2, 1)
cost = 20 + 4.5 * distance + 25 * (attempts - 1) + rng.normal(0, 5, N)
capacity = rng.choice([40, 60, 150], N)
load = np.minimum(capacity, parcels * rng.uniform(0.8, 2.5, N))

df = pd.DataFrame({
    "order_id": np.arange(1, N + 1), "area": area, "latitude": lat, "longitude": lon,
    "distance_km": distance.round(2), "dispatch_time": dispatch, "promised_time": promised,
    "delivery_time": delivery, "attempts": attempts, "total_cost": cost.round(2),
    "load_kg": load.round(1), "vehicle_capacity_kg": capacity, "rain_mm": rain.round(1),
    "parcels_on_route": parcels,
})

# Introduce data-quality problems on purpose
df.loc[rng.choice(N, 90, replace=False), "distance_km"] = np.nan
df.loc[rng.choice(N, 40, replace=False), "rain_mm"] = np.nan
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("deliveries.csv", index=False)
print("Saved deliveries.csv with", len(df), "rows")
