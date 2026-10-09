"""Week 2: Data cleaning and preprocessing pipeline for QuickCart Chennai.

Input : deliveries_raw.csv   (messy data, see inject_quality_issues.py)
Output: deliveries_clean_w2.csv    (clean, original units)
        deliveries_model_ready.csv (scaled + encoded, ready for modeling)
        results.json               (counts used in the report)
        fig1_issues.png, fig2_distance_outliers.png, fig3_normalization.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

R = {}  # results log

# ------------------------------------------------------------------ 1. LOAD + PROFILE
raw = pd.read_csv("deliveries_raw.csv")
R["raw_rows"] = len(raw)
R["raw_cols"] = raw.shape[1]
R["missing_raw"] = raw.isnull().sum()[lambda s: s > 0].to_dict()
R["dup_rows"] = int(raw.duplicated(subset="order_id").sum())
R["area_labels_raw"] = int(raw["area"].nunique())
R["neg_cost"] = int((raw["total_cost"] < 0).sum())
R["dist_gt60"] = int((raw["distance_km"] > 60).sum())
d_t = pd.to_datetime(raw["dispatch_time"], format="ISO8601")
v_t = pd.to_datetime(raw["delivery_time"], format="ISO8601")
R["time_travel"] = int((v_t < d_t).sum())

# "Naive" KPI values on the raw data (what a careless analyst would report)
R["raw_avg_distance"] = round(raw["distance_km"].mean(), 2)
R["raw_avg_cost"] = round(raw["total_cost"].mean(), 2)
R["raw_avg_minutes"] = round(((v_t - d_t).dt.total_seconds() / 60).mean(), 2)
R["raw_max_distance"] = round(raw["distance_km"].max(), 1)

df = raw.copy()

# ------------------------------------------------------------------ 2. DUPLICATES
df = df.drop_duplicates(subset="order_id").copy()
R["rows_after_dedup"] = len(df)

# ------------------------------------------------------------------ 3. TEXT STANDARDIZATION
canonical = {"avadi": "Avadi", "ambattur": "Ambattur", "ambatur": "Ambattur",
             "tambaram": "Tambaram", "velachery": "Velachery", "annanagar": "Anna Nagar",
             "adyar": "Adyar", "porur": "Porur", "perungudi": "Perungudi"}
key = df["area"].str.lower().str.replace(r"[^a-z]", "", regex=True)
df["area"] = key.map(canonical)
R["area_unmapped"] = int(df["area"].isnull().sum())
R["area_labels_clean"] = int(df["area"].nunique())

# ------------------------------------------------------------------ 4. DATES + INVALID ROWS
df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="ISO8601")
df["promised_time"] = pd.to_datetime(df["promised_time"], format="ISO8601")
df["delivery_time"] = pd.to_datetime(df["delivery_time"], format="ISO8601")

R["missing_delivery_time"] = int(df["delivery_time"].isnull().sum())
df = df.dropna(subset=["delivery_time"])            # target cannot be imputed honestly
R["rows_after_drop_missing_time"] = len(df)

bad_time = df["delivery_time"] < df["dispatch_time"]
R["time_travel_after_dedup"] = int(bad_time.sum())
df = df[~bad_time].copy()
R["rows_after_drop_bad_time"] = len(df)

# ------------------------------------------------------------------ 5. NEGATIVE COSTS
R["neg_cost_after_dedup"] = int((df["total_cost"] < 0).sum())
df["total_cost"] = df["total_cost"].abs()           # sign typo: magnitudes look normal

# ------------------------------------------------------------------ 6. OUTLIER DETECTION (distance)
def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

before = df["distance_km"].copy()
lo, hi = iqr_bounds(before.dropna())
R["iqr_lo"], R["iqr_hi"] = round(lo, 2), round(hi, 2)
R["iqr_outliers"] = int(((before < lo) | (before > hi)).sum())
z = (before - before.mean()) / before.std()
R["z_outliers"] = int((z.abs() > 3).sum())

# Domain rule: no delivery inside the Chennai service area is > 60 km from its hub
impossible = df["distance_km"] > 60
R["impossible_dist"] = int(impossible.sum())
R["impossible_max"] = round(df.loc[impossible, "distance_km"].max(), 1)
df.loc[impossible, "distance_km"] = np.nan          # treat as missing, re-impute below
R["legit_long_kept"] = int(((df["distance_km"] > hi)).sum())   # genuine long trips stay

# ------------------------------------------------------------------ 7. MISSING VALUES
R["missing_before_impute"] = df[["distance_km", "rain_mm", "load_kg"]].isnull().sum().to_dict()
dist_mean_before_imp = round(df["distance_km"].mean(), 3)
dist_std_before_imp = round(df["distance_km"].std(), 3)

df["distance_km"] = df.groupby("area")["distance_km"].transform(lambda x: x.fillna(x.median()))
df["rain_mm"] = df["rain_mm"].fillna(0)
df["load_kg"] = df.groupby("vehicle_capacity_kg")["load_kg"].transform(lambda x: x.fillna(x.median()))

R["dist_mean_before_imp"], R["dist_std_before_imp"] = dist_mean_before_imp, dist_std_before_imp
R["dist_mean_after_imp"] = round(df["distance_km"].mean(), 3)
R["dist_std_after_imp"] = round(df["distance_km"].std(), 3)
R["missing_after"] = int(df.isnull().sum().sum())

# ------------------------------------------------------------------ 8. FEATURE ENGINEERING
df["delivery_minutes"] = (df["delivery_time"] - df["dispatch_time"]).dt.total_seconds() / 60
df["hour"] = df["dispatch_time"].dt.hour
df["day_of_week"] = df["dispatch_time"].dt.dayofweek
df["is_peak_hour"] = df["hour"].isin([8, 9, 17, 18, 19]).astype(int)
df["is_late"] = (df["delivery_time"] > df["promised_time"]).astype(int)

# ------------------------------------------------------------------ 9. VALIDATION CHECKS
assert df.isnull().sum().sum() == 0
assert not df["order_id"].duplicated().any()
assert (df["total_cost"] >= 0).all()
assert (df["delivery_time"] >= df["dispatch_time"]).all()
assert df["distance_km"].between(0, 60).all()
R["clean_rows"] = len(df)
R["rows_removed"] = R["raw_rows"] - R["clean_rows"]
R["rows_removed_pct"] = round(100 * R["rows_removed"] / R["raw_rows"], 2)

R["clean_avg_distance"] = round(df["distance_km"].mean(), 2)
R["clean_avg_cost"] = round(df["total_cost"].mean(), 2)
R["clean_avg_minutes"] = round(df["delivery_minutes"].mean(), 2)
R["clean_on_time"] = round((1 - df["is_late"].mean()) * 100, 1)
R["clean_max_distance"] = round(df["distance_km"].max(), 1)

df.to_csv("deliveries_clean_w2.csv", index=False)

# ------------------------------------------------------------------ 10. NORMALIZATION + ENCODING
cols = ["distance_km", "rain_mm", "parcels_on_route", "load_kg", "total_cost", "delivery_minutes"]
mm = MinMaxScaler().fit(df[cols])
zs = StandardScaler().fit(df[cols])
mm_df = pd.DataFrame(mm.transform(df[cols]), columns=[c + "_minmax" for c in cols], index=df.index)
zs_df = pd.DataFrame(zs.transform(df[cols]), columns=[c + "_z" for c in cols], index=df.index)

R["norm_table"] = {c: {"min": round(df[c].min(), 2), "max": round(df[c].max(), 2),
                       "mean": round(df[c].mean(), 2), "std": round(df[c].std(), 2),
                       "mm_min": round(mm_df[c + "_minmax"].min(), 2),
                       "mm_max": round(mm_df[c + "_minmax"].max(), 2),
                       "z_mean": round(zs_df[c + "_z"].mean(), 2),
                       "z_std": round(zs_df[c + "_z"].std(), 2)} for c in cols}

area_dummies = pd.get_dummies(df["area"], prefix="area", dtype=int)
model_ready = pd.concat([df[["order_id", "is_peak_hour", "hour", "day_of_week", "is_late"]],
                         mm_df, zs_df, area_dummies], axis=1)
model_ready.to_csv("deliveries_model_ready.csv", index=False)
R["model_ready_cols"] = model_ready.shape[1]
R["area_dummy_cols"] = area_dummies.shape[1]

# ------------------------------------------------------------------ 11. FIGURES
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAVY, BLUE, RED, GREY = "#1F3864", "#2E5597", "#C0392B", "#7F8C8D"

# Fig 1: issues found
issues = {
    "Duplicate orders": R["dup_rows"],
    "Inconsistent area labels\n(extra labels)": R["area_labels_raw"] - 8,
    "Missing delivery time": R["missing_delivery_time"],
    "Missing distance": R["missing_raw"].get("distance_km", 0),
    "Missing rainfall": R["missing_raw"].get("rain_mm", 0),
    "Missing load": R["missing_raw"].get("load_kg", 0),
    "Delivery before dispatch": R["time_travel"],
    "Impossible distance (>60 km)": R["dist_gt60"],
    "Negative cost": R["neg_cost"],
}
fig, ax = plt.subplots(figsize=(7.5, 4))
items = sorted(issues.items(), key=lambda t: t[1])
ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE)
for i, (_, v) in enumerate(items):
    ax.text(v + 1, i, str(v), va="center")
ax.set_xlabel("Number of records affected")
ax.set_title("Data quality issues found in the raw dataset", color=NAVY, fontweight="bold")
plt.tight_layout(); plt.savefig("fig1_issues.png", dpi=170); plt.close()

# Fig 2: distance before vs after
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
axes[0].boxplot(raw["distance_km"].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor="#F5B7B1"), medianprops=dict(color=RED))
axes[0].set_title("Before cleaning", color=RED); axes[0].set_ylabel("Distance (km)")
axes[1].boxplot(df["distance_km"], vert=True, patch_artist=True,
                boxprops=dict(facecolor="#AED6F1"), medianprops=dict(color=NAVY))
axes[1].set_title("After cleaning", color=NAVY)
fig.suptitle("Distance outliers removed (max %.0f km to %.1f km)" % (R["raw_max_distance"], R["clean_max_distance"]),
             color=NAVY, fontweight="bold")
plt.tight_layout(); plt.savefig("fig2_distance_outliers.png", dpi=170); plt.close()

# Fig 3: normalization
fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
axes[0].hist(df["distance_km"], bins=30, color=GREY); axes[0].set_title("Original (km)")
axes[1].hist(mm_df["distance_km_minmax"], bins=30, color=BLUE); axes[1].set_title("Min-Max (0 to 1)")
axes[2].hist(zs_df["distance_km_z"], bins=30, color=NAVY); axes[2].set_title("Z-score (mean 0, sd 1)")
for a in axes: a.set_yticks([])
fig.suptitle("Same distribution shape, different scale", color=NAVY, fontweight="bold")
plt.tight_layout(); plt.savefig("fig3_normalization.png", dpi=170); plt.close()

json.dump(R, open("results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
