"""Week 2, Step 1: Simulate a messy 'raw' dataset.

Starts from deliveries.csv (Week 1 sample data, which already contains
missing distances, missing rainfall and duplicate rows) and adds the kinds of
errors that real logistics data usually has:
  - inconsistent area spellings (case, spaces, typos)
  - impossible distances (data-entry errors)
  - delivery time earlier than dispatch time (clock / entry errors)
  - negative costs
  - missing delivery timestamps and missing load values
Saves deliveries_raw.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
df = pd.read_csv("deliveries.csv")
n = len(df)

# 1. Inconsistent area names
idx = rng.choice(n, 160, replace=False)
variants = {
    "Velachery": ["velachery", "VELACHERY", "Velachery "],
    "Tambaram": ["tambaram", "Tambaram ", "Tambaram."],
    "Anna Nagar": ["anna nagar", "Anna nagar", "AnnaNagar"],
    "Ambattur": ["ambattur", "Ambatur"],
    "Avadi": ["avadi", "AVADI"],
}
for i in idx:
    a = df.at[i, "area"]
    if a in variants:
        df.at[i, "area"] = rng.choice(variants[a])

# 2. Impossible distances (entry errors, e.g. metres typed as km)
idx = rng.choice(n, 12, replace=False)
df.loc[idx, "distance_km"] = rng.uniform(150, 900, 12).round(1)

# 3. Delivery time earlier than dispatch time
idx = rng.choice(n, 14, replace=False)
dt = pd.to_datetime(df.loc[idx, "dispatch_time"]) - pd.to_timedelta(rng.integers(5, 90, 14), unit="m")
df.loc[idx, "delivery_time"] = dt.astype(str)

# 4. Negative costs
idx = rng.choice(n, 6, replace=False)
df.loc[idx, "total_cost"] = -df.loc[idx, "total_cost"]

# 5. Missing delivery timestamps and load values
df.loc[rng.choice(n, 30, replace=False), "delivery_time"] = np.nan
df.loc[rng.choice(n, 55, replace=False), "load_kg"] = np.nan

df.to_csv("deliveries_raw.csv", index=False)
print("Saved deliveries_raw.csv:", df.shape)
