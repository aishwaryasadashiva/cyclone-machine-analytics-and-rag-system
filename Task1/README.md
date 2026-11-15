# Task 1 — Cyclone machine data analysis

## Run
```bash
pip install -r requirements.txt            # Python 3.11+ recommended
python task1_analysis.py /path/to/data.xlsx
```
`data.xlsx` is not included; pass its path (otherwise the script looks next to itself, one folder up, then the current folder). Everything is written next to the script, and `plots/` is created. Runtime is about 2–3 minutes after the Excel file is read; the first read takes about a minute longer and is cached in `.raw_cache.pkl` (safe to delete). Seeds are fixed (42).

Numbers here come from Python 3.12, pandas 3.0.6, numpy 2.5.4 and scikit-learn 1.9.1. On pandas 2.2.3 / numpy 1.26.4 / scikit-learn 1.4.2 the only differences were 549 instead of 550 anomaly events and a boosting RMSE of 15.29 instead of 15.28.

## Required deliverables
| File | Content |
|---|---|
| `task1_analysis.py` | Full pipeline: clean, EDA, shutdowns, clustering, anomalies and root causes, forecasting, insights |
| `shutdown_periods.csv` | 148 shutdown / idle periods: start, end, duration, minimum inlet temperature, type |
| `clusters_summary.csv` | Per state: mean, std, p05, p50, p95 per variable, share of time, episodes, episode durations, description |
| `anomalous_periods.csv` | 550 anomaly events: start, end, duration, state, implicated variables, minutes to next shutdown |
| `forecasts.csv` | True vs predicted for 4 methods at 5–60 min ahead |
| `plots/` | 22 PNG files (EDA, shutdowns, clusters, anomalies, forecasts) |

## Additional outputs
| File | Content |
|---|---|
| `insights.md`, `insights_numbers.json` | Generated insights, recommendations and the numbers behind them (edit wording in the script, not here) |
| `root_cause_hypotheses.csv` | 5 selected anomalies with evidence and a hypothesis (plots `10_anomaly_1..5.png`) |
| `forecast_metrics.csv`, `forecast_intervals.csv`, `next_1_hour_forecast.csv` | RMSE / MAE per model, horizon and regime; 10th–90th percentile band; 12-step forecast from the last observation |
| `summary_statistics.csv`, `correlation_matrix.csv`, `data_quality_report.csv` | Summary statistics, correlations, cleaning counts |
| `yearly_drift.csv`, `state_stability_by_year.csv` | Per-year behaviour and state stability |
| `anomaly_crosscheck_*.csv` | Second detector (robust Mahalanobis) as a consistency check |

## Method in brief
1. **Preparation:** text codes and sensor limits become NaN, strict 5-minute grid (861 missing timestamps added), isolated spikes removed (Hampel rule), gaps up to 30 min interpolated.
2. **Shutdowns:** median of the three drafts above -60 for at least 30 min, with stops under 30 min apart merged.
3. **States:** KMeans (k = 5) on active data, using trailing-window features only.
4. **Anomalies:** one Isolation Forest per state, merged into events.
5. **Forecast:** inlet gas temperature, 12 steps (1 hour); persistence, moving average, Ridge autoregression and gradient boosting; chronological 80/20 split.

## Key results
148 shutdown / idle events (6,877 h, 21.8 % of the record); 5 states; 550 anomaly events. At 60 min ahead, gradient boosting reaches RMSE 17.2 / MAE 9.7 against 21.4 / 12.4 for persistence. Details and recommendations are in `insights.md`.

## Caveats
- No shutdown labels or maintenance logs were provided, so the shutdown rule, state names and root causes are interpretations to validate with operators.
- The states are not stable over time (adjusted Rand index 0.44 between the first and second half of the record).
- ARIMA and Prophet were not used; Ridge autoregression is the linear comparator. Only one chronological split was evaluated.
