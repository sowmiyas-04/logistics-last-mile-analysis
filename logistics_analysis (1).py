"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Simulates a last-mile logistics dataset, runs EDA, and produces 8 charts.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold", "axes.titlesize": 13})
OUT = "charts/"

# ------------------------------------------------------------------
# 1. DATA SIMULATION
# ------------------------------------------------------------------
N = 5000
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, N), unit="D")
hub = np.random.choice(["Chennai", "Bengaluru", "Hyderabad", "Mumbai", "Delhi"], N,
                       p=[0.24, 0.22, 0.18, 0.20, 0.16])
vehicle = np.random.choice(["Bike", "Van", "Truck"], N, p=[0.40, 0.42, 0.18])
hour = np.clip(np.random.normal(13, 3.6, N).round(), 8, 20).astype(int)

dist_mean = {"Bike": 6, "Van": 18, "Truck": 55}
distance = np.array([max(1.0, np.random.gamma(2.2, dist_mean[v] / 2.2)) for v in vehicle])

w_mean = {"Bike": 4, "Van": 45, "Truck": 320}
weight = np.array([max(0.5, np.random.gamma(2.0, w_mean[v] / 2.0)) for v in vehicle])
units = np.maximum(1, (weight / np.random.uniform(1.5, 4.0, N)).round()).astype(int)

month = dates.month
weather = np.where(np.isin(month, [10, 11, 12]) & (np.random.rand(N) < 0.35), "Rain",
                   np.where(np.random.rand(N) < 0.08, "Rain", "Clear"))
peak = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 19))
traffic = np.where(peak & (np.random.rand(N) < 0.70), "High",
                   np.where(np.random.rand(N) < 0.30, "Medium", "Low"))

speed_base = {"Bike": 24, "Van": 30, "Truck": 38}
speed = np.array([speed_base[v] for v in vehicle], dtype=float)
speed *= np.where(traffic == "High", 0.60, np.where(traffic == "Medium", 0.82, 1.0))
speed *= np.where(weather == "Rain", 0.80, 1.0)
speed *= np.random.normal(1, 0.08, N)

service_time = np.random.normal(0.25, 0.08, N).clip(0.05)           # hrs at doorstep
sort_delay = np.where(hub == "Mumbai", np.random.exponential(0.55, N),
              np.where(hub == "Delhi", np.random.exponential(0.40, N),
                       np.random.exponential(0.22, N)))
delivery_time = distance / speed + service_time + sort_delay

cost_per_km = {"Bike": 4.5, "Van": 11.0, "Truck": 24.0}
fuel_cost = np.array([cost_per_km[v] for v in vehicle]) * distance * np.random.normal(1, 0.10, N)
handling = 6 + 0.12 * weight + np.random.normal(0, 3, N)
labour = delivery_time * np.array([{"Bike": 90, "Van": 130, "Truck": 180}[v] for v in vehicle])
cost = (fuel_cost + handling + labour).clip(15)

promised = {"Bike": 1.2, "Van": 2.2, "Truck": 4.5}
sla = np.array([promised[v] for v in vehicle])
delayed = (delivery_time > sla).astype(int)
rating = (5 - 1.4 * (delivery_time / sla) + np.random.normal(0, 0.5, N)).clip(1, 5).round(1)

df = pd.DataFrame({
    "delivery_id": [f"D{100000+i}" for i in range(N)],
    "date": dates, "month": month, "weekday": dates.dayofweek, "hour": hour,
    "hub": hub, "vehicle": vehicle, "distance_km": distance.round(2),
    "weight_kg": weight.round(1), "units": units, "traffic": traffic, "weather": weather,
    "delivery_time_hr": delivery_time.round(2), "fuel_cost": fuel_cost.round(2),
    "total_cost": cost.round(2), "sla_hr": sla, "delayed": delayed, "rating": rating,
})
df["cost_per_km"] = (df.total_cost / df.distance_km).round(2)
df.to_csv("logistics_dataset.csv", index=False)

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
S = {}
S["n"] = N
S["missing"] = int(df.isna().sum().sum())
num_cols = ["distance_km", "weight_kg", "units", "delivery_time_hr", "total_cost", "cost_per_km", "rating"]
desc = df[num_cols].describe().T
desc["median"] = df[num_cols].median()
desc["skew"] = df[num_cols].skew()
S["desc"] = desc.round(2).to_dict("index")
S["delay_rate"] = round(df.delayed.mean() * 100, 1)
S["corr"] = df[num_cols + ["delayed"]].corr().round(2).to_dict()

S["by_vehicle"] = df.groupby("vehicle").agg(
    n=("delivery_id", "count"), mean_time=("delivery_time_hr", "mean"), mean_cost=("total_cost", "mean"),
    cost_km=("cost_per_km", "mean"), delay=("delayed", "mean"), dist=("distance_km", "mean")
).round(2).to_dict("index")
S["by_hub"] = df.groupby("hub").agg(
    n=("delivery_id", "count"), mean_time=("delivery_time_hr", "mean"), mean_cost=("total_cost", "mean"),
    delay=("delayed", "mean")).round(3).to_dict("index")
S["by_traffic"] = df.groupby("traffic").agg(n=("delivery_id", "count"), mean_time=("delivery_time_hr", "mean"),
    delay=("delayed", "mean")).round(3).to_dict("index")
S["by_weather"] = df.groupby("weather").agg(n=("delivery_id", "count"), mean_time=("delivery_time_hr", "mean"),
    delay=("delayed", "mean")).round(3).to_dict("index")
S["by_month"] = df.groupby("month").agg(n=("delivery_id", "count"), mean_time=("delivery_time_hr", "mean"),
    delay=("delayed", "mean")).round(3).to_dict("index")
peak_mask = ((df.hour >= 8) & (df.hour <= 10)) | ((df.hour >= 17) & (df.hour <= 19))
S["peak_delay"] = round(df[peak_mask].delayed.mean() * 100, 1)
S["offpeak_delay"] = round(df[~peak_mask].delayed.mean() * 100, 1)
S["peak_share"] = round(peak_mask.mean() * 100, 1)
S["peak_delayed_share"] = round(df[peak_mask].delayed.sum() / df.delayed.sum() * 100, 1)

# Outliers (IQR) on delivery time
q1, q3 = df.delivery_time_hr.quantile([.25, .75])
iqr = q3 - q1
S["outliers"] = int((df.delivery_time_hr > q3 + 1.5 * iqr).sum())
S["outlier_threshold"] = round(q3 + 1.5 * iqr, 2)

# Cost drivers: regression of total cost on distance, weight
slope, intercept, r, p, se = stats.linregress(df.distance_km, df.total_cost)
S["reg_dist_cost"] = {"slope": round(slope, 2), "intercept": round(intercept, 2), "r2": round(r ** 2, 3)}
# Rating vs delay
S["rating_delayed"] = round(df[df.delayed == 1].rating.mean(), 2)
S["rating_ontime"] = round(df[df.delayed == 0].rating.mean(), 2)
# ANOVA across hubs for delivery time-per-km normalised
df["time_per_km"] = df.delivery_time_hr / df.distance_km
F, pval = stats.f_oneway(*[g.delay_ if False else g.delivery_time_hr.values for _, g in df[df.vehicle == "Van"].groupby("hub")])
S["anova_van_hub"] = {"F": round(F, 2), "p": float(f"{pval:.2e}")}
# Cost share
S["cost_share_vehicle"] = (df.groupby("vehicle").total_cost.sum() / df.total_cost.sum() * 100).round(1).to_dict()
S["delay_share_vehicle"] = (df.groupby("vehicle").delayed.sum() / df.delayed.sum() * 100).round(1).to_dict()
S["total_cost"] = round(df.total_cost.sum(), 0)

# ------------------------------------------------------------------
# 3. VISUALISATIONS
# ------------------------------------------------------------------
# Fig 1 - distribution of delivery time
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(df.delivery_time_hr, bins=50, kde=True, color="#2a6f97", ax=ax)
ax.axvline(df.delivery_time_hr.mean(), color="crimson", ls="--", label=f"Mean = {df.delivery_time_hr.mean():.2f} h")
ax.axvline(df.delivery_time_hr.median(), color="darkgreen", ls="--", label=f"Median = {df.delivery_time_hr.median():.2f} h")
ax.set(title="Fig 1. Distribution of Delivery Time", xlabel="Delivery time (hours)", ylabel="Number of deliveries")
ax.legend(); plt.tight_layout(); plt.savefig(OUT + "fig1_hist.png"); plt.close()

# Fig 2 - box plot by vehicle
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=df, x="vehicle", y="delivery_time_hr", order=["Bike", "Van", "Truck"], hue="vehicle",
            palette="Set2", legend=False, ax=ax)
ax.set(title="Fig 2. Delivery Time by Vehicle Type", xlabel="Vehicle", ylabel="Delivery time (hours)")
plt.tight_layout(); plt.savefig(OUT + "fig2_box_vehicle.png"); plt.close()

# Fig 3 - correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
cm = df[num_cols + ["delayed"]].corr()
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Fig 3. Correlation Matrix of Key Metrics")
plt.tight_layout(); plt.savefig(OUT + "fig3_corr.png"); plt.close()

# Fig 4 - scatter distance vs cost
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="total_cost", hue="vehicle",
                hue_order=["Bike", "Van", "Truck"], alpha=0.55, s=22, ax=ax)
ax.set(title="Fig 4. Distance vs Transportation Cost (1,500-delivery sample)",
       xlabel="Distance (km)", ylabel="Total cost (INR)")
plt.tight_layout(); plt.savefig(OUT + "fig4_scatter.png"); plt.close()

# Fig 5 - monthly trend (dual axis)
m = df.groupby("month").agg(vol=("delivery_id", "count"), t=("delivery_time_hr", "mean"),
                            d=("delayed", "mean")).reset_index()
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(m.month, m.vol, color="#a9d6e5", label="Shipments")
ax1.set_xlabel("Month (2025)"); ax1.set_ylabel("Shipments")
ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(m.month, m.d * 100, color="crimson", marker="o", label="Delay rate (%)")
ax2.set_ylabel("Delay rate (%)"); ax2.grid(False)
ax1.set_title("Fig 5. Monthly Shipment Volume and Delay Rate")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")
plt.tight_layout(); plt.savefig(OUT + "fig5_trend.png"); plt.close()

# Fig 6 - delay by hub
h = df.groupby("hub").delayed.mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=h.index, y=h.values, hue=h.index, palette="rocket", legend=False, ax=ax)
for i, v in enumerate(h.values):
    ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=10)
ax.set(title="Fig 6. SLA Delay Rate by Hub", xlabel="Hub", ylabel="Deliveries late (%)")
plt.tight_layout(); plt.savefig(OUT + "fig6_hub_delay.png"); plt.close()

# Fig 7 - heatmap hour x weekday delay
pv = df.pivot_table(index="weekday", columns="hour", values="delayed", aggfunc="mean") * 100
pv.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.heatmap(pv, cmap="YlOrRd", annot=False, cbar_kws={"label": "Delay rate (%)"}, ax=ax)
ax.set(title="Fig 7. Delay Rate by Hour of Day and Weekday", xlabel="Hour of day", ylabel="")
plt.tight_layout(); plt.savefig(OUT + "fig7_heatmap.png"); plt.close()

# Fig 8 - cost per km by vehicle and traffic
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=df, x="vehicle", y="cost_per_km", hue="traffic", order=["Bike", "Van", "Truck"],
            hue_order=["Low", "Medium", "High"], palette="YlOrBr", errorbar=None, ax=ax)
ax.set(title="Fig 8. Cost per km by Vehicle Type and Traffic Level", xlabel="Vehicle", ylabel="Cost per km (INR)")
plt.tight_layout(); plt.savefig(OUT + "fig8_costkm.png"); plt.close()

json.dump(S, open("stats.json", "w"), indent=1, default=float)
print(json.dumps(S, indent=1, default=float))
