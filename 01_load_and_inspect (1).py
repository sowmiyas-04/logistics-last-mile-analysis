"""Step 1: Load the delivery data and take a first look."""
import pandas as pd

df = pd.read_csv("deliveries.csv", parse_dates=["dispatch_time", "promised_time", "delivery_time"])

print("Shape (rows, columns):", df.shape)
print("\nColumn types and non-null counts:")
df.info()
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate order_ids:", df["order_id"].duplicated().sum())
print("\nStatistical summary:")
print(df.describe())
