# Week 4: Predictive Modeling and Optimization in Logistics Systems

Forecasts last-mile **delivery time (minutes)** on a simulated dataset of 6,000 deliveries,
then uses the forecasts to optimise dispatch, promise times and vehicle allocation.
The data is simulated (no real delivery log was available); see Section 10 of the report.

## Files
| File | Purpose |
|---|---|
| `logistics_model.py` | Simulates data, trains 5 models, 5-fold CV, hyperparameter tuning, statistical tests, residual diagnostics, first optimisation pass |
| `optim_v2.py` | Final optimisation: 90-min SLA analysis, conformal P90 promise times, vehicle-allocation linear program, peak-shifting and route-density what-ifs |
| `finalize.py` | Chi-square tests at the 90-min SLA and the final optimisation figure |
| `simulated_deliveries.csv` | The generated dataset |
| `results.json`, `results2.json`, `results3.json` | Every number quoted in the report |
| `figures/` | Charts used in the report |

## How to run
```bash
pip install -r requirements.txt
python logistics_model.py   # slow on one CPU core (several minutes); writes the CSV and results.json
python optim_v2.py          # needs the CSV; writes results2.json
python finalize.py          # needs results.json and results2.json; writes results3.json
```
Run in that order. Random seeds are fixed (42 for data, split, models and search; 1 for the
calibration split; 7 for the what-if sampling).
Note: `optim_v2.py` and `finalize.py` write charts to a `fig/` folder, so create it first (`mkdir fig`).

## Headline results (test set, 1,200 deliveries)
- Tuned Gradient Boosting: RMSE 10.07 min (95% CI 9.51 to 10.70), MAE 7.46 min, R2 0.873
- Linear regression baseline: RMSE 11.37 min
- Urban peak deliveries: 29% of orders, 55% of deliveries over 90 min
- Vehicle-allocation LP: about 20% lower cost (predicted, assumed cost rates)
