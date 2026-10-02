"""Step 3: Calculate the five KPIs."""
import pandas as pd

df = pd.read_csv("deliveries_clean.csv")

on_time_rate = (1 - df["is_late"].mean()) * 100
avg_delivery_time = df["delivery_minutes"].mean()
cost_per_delivery = df["total_cost"].sum() / len(df)
first_attempt_ok = (df["attempts"] == 1).mean() * 100
vehicle_util = (df["load_kg"] / df["vehicle_capacity_kg"]).mean() * 100

print(f"On-Time Delivery Rate     : {on_time_rate:.1f}%")
print(f"Average Delivery Time     : {avg_delivery_time:.1f} min")
print(f"Cost per Delivery         : Rs {cost_per_delivery:.2f}")
print(f"First-Attempt Success Rate: {first_attempt_ok:.1f}%")
print(f"Vehicle Utilization       : {vehicle_util:.1f}%")
