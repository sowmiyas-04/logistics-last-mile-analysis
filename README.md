# Last-Mile Delivery Optimization for an E-Commerce Company in Chennai

Logistics Data Analyst Intern project, YuvaIntern virtual internship (Week 1: Strategic Planning and Data Exploration).

## Scenario
QuickCart Chennai, a hypothetical e-commerce company, delivers about 3,000 parcels a day from three hubs using two-wheelers and light vans. It faces late deliveries, unbalanced routes, failed first attempts and poorly used vehicles. This project plans how data science with Python can address these problems.

## KPIs
1. On-Time Delivery Rate
2. Average Delivery Time
3. Cost per Delivery
4. First-Attempt Success Rate
5. Vehicle Utilization

## Techniques
Exploratory analysis, regression (delivery time prediction), K-Means clustering (delivery zones), and route optimization (Vehicle Routing Problem).

## Files
| File | Purpose |
|---|---|
| `generate_sample_data.py` | Creates a simulated dataset `deliveries.csv` |
| `01_load_and_inspect.py` | Loads and inspects the data |
| `02_clean_and_features.py` | Cleaning and feature engineering |
| `03_kpis.py` | Calculates the five KPIs |
| `04_clustering_zones.py` | K-Means delivery zones |
| `05_delivery_time_model.py` | Random Forest delivery time model |
| `06_route_optimization_pseudocode.md` | Route optimization plan |

## How to run
```
pip install -r requirements.txt
python generate_sample_data.py
python 01_load_and_inspect.py
python 02_clean_and_features.py
python 03_kpis.py
python 04_clustering_zones.py
python 05_delivery_time_model.py
```

## Note
The data is simulated for learning purposes. Public datasets such as the Amazon Last Mile Routing Research Challenge dataset can be used in later weeks.

## Author
Sowmiya
