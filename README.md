# ⚡ Energy Forecasting & Anomaly Detection API

A FastAPI service that serves the trained models of the **AI-Powered Energy Consumption Forecasting & Anomaly Detection System**. Send a week of hourly household electricity usage (kW) and get back:

- a **forecast** for 1 hour ahead or 24 hours ahead, and
- an **anomaly check** for a measured hourly value (spike or drop).

**Live API (Swagger docs):** https://energy-api-project-d1d7b.containers.snapdeploy.app/docs

## Part of a bigger project

This repository contains only the API. The data cleaning, EDA, model training, MLflow tracking and the Streamlit dashboard live in the main project repository:

👉 [AI-Powered-Energy-Consumption-Forecasting-Anomaly-Detection-System](https://github.com/Baloxh-Aziz/AI-Powered-Energy-Consumption-Forecasting-Anomaly-Detection-System)

- [Live dashboard](https://household-energy-forecasting-and-anomaly-detection.streamlit.app) (it calls this API for its live demo)

The models here are exported by notebook `06 model export` of that repository.

## Endpoints

| Method | Path              | Description                                              |
|--------|-------------------|----------------------------------------------------------|
| GET    | `/health`         | Check that the API is alive and see which models loaded  |
| POST   | `/predict`        | Forecast usage 1 hour or 24 hours ahead                  |
| POST   | `/detect-anomaly` | Check whether an actual hourly value is an anomaly       |

Interactive docs (pre-filled examples, one-click "Try it out") are available at `/docs`.

### `POST /predict`

| Field         | Type          | Description                                                        |
|---------------|---------------|--------------------------------------------------------------------|
| `history`     | list of float | Hourly usage in kW, oldest first, **at least 168 values** (1 week) |
| `target_time` | string        | The hour to forecast, e.g. `"2024-07-01 12:00:00"`                 |
| `horizon`     | string        | `"1h"` (default) or `"24h"`                                        |

Response:

```json
{
  "target_time": "2024-07-01 12:00:00",
  "horizon": "1h",
  "forecast_kw": 0.8123
}
```

### `POST /detect-anomaly`

| Field         | Type          | Description                                                                          |
|---------------|---------------|--------------------------------------------------------------------------------------|
| `history`     | list of float | Hourly usage up to the hour **before** `target_time`, at least 168 values            |
| `target_time` | string        | The hour to check                                                                    |
| `actual`      | float         | Measured usage (kW) in that hour                                                     |
| `k`           | float, optional | Sensitivity. Lower = more alerts. Defaults to the value saved with the model (5)   |

Response:

```json
{
  "target_time": "2024-07-01 12:00:00",
  "actual_kw": 4.5,
  "expected_kw": 0.8123,
  "z_score": 12.4,
  "k": 5.0,
  "is_anomaly": true,
  "type": "spike"
}
```

### Errors

The API returns `400` with a clear message if:

- `history` has fewer than 168 values or contains missing / invalid numbers,
- `target_time` cannot be parsed,
- `horizon` is not `"1h"` or `"24h"`.

## Example usage

The `history` list needs 168 hourly values. This Python snippet builds a dummy week and calls the live API:

```python
import numpy as np
import requests

URL = "https://energy-api-project-d1d7b.containers.snapdeploy.app"

history = [round(0.7 + 0.5 * np.sin(i / 24 * 2 * np.pi), 3) for i in range(168)]

# 1) forecast
r = requests.post(f"{URL}/predict", json={
    "history": history,
    "target_time": "2024-07-01 12:00:00",
    "horizon": "1h",
})
print(r.json())

# 2) anomaly check
r = requests.post(f"{URL}/detect-anomaly", json={
    "history": history,
    "target_time": "2024-07-01 12:00:00",
    "actual": 4.5,
})
print(r.json())
```

Health check:

```bash
curl https://energy-api-project-d1d7b.containers.snapdeploy.app/health
```

## How it works

- **Forecasting:** Random Forest models that predict the *change* from a base lag (delta target) and then add the base value back. Features: hour, weekday, weekend flag, lags (1, 2, 24, 48, 168 h) and, for the 1-hour model, rolling mean/std of the past hours. All features use past data only, so there is no leakage.
- **Anomaly detection:** the 1-hour model gives an expected value; the relative residual is turned into a robust z-score (median / MAD). If `|z| > k` the hour is flagged as a `spike` or `drop`. The threshold was calibrated on a validation period, not on the test set.

For the full evaluation (baselines, XGBoost, LightGBM, LSTM, weather experiment, per-city results) see the main project repository.

## Run locally

```bash
git clone https://github.com/Baloxh-Aziz/energy-api-project.git
cd energy-api-project
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs.

The `.pkl` model files are loaded from the current folder by default. To load them from somewhere else, set the `MODEL_DIR` environment variable:

```bash
MODEL_DIR=/path/to/models uvicorn main:app --reload
```

## Repository structure

```
main.py                   FastAPI app (endpoints, feature building, prediction logic)
energy_forecast_1h.pkl    Random Forest model, 1 hour ahead
energy_forecast_24h.pkl   Random Forest model, 24 hours ahead
anomaly_model.pkl         Anomaly bundle (model, median, sigma, default k)
requirements.txt          API dependencies
```

## Limitations

- The deployed models were trained on **one house (Karachi_38)** and serve as a showcase, not a general model for every house.
- At least one week (168 hours) of history is required for every request.
- The detector flags statistical outliers (point anomalies), not confirmed faults. Holidays such as Eid can look like anomalies.

## Tech stack

Python, FastAPI, Uvicorn, Pandas, NumPy, scikit-learn 1.6.1, Joblib.

## Data

[REWDP: Residential Energy and Weather Data, Pakistan (LUMS)](https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html), minute-level smart-meter readings, resampled to hourly values.

## Group Members

Azizullah Asad & Muhammad Yahya. Built as the final project of the SMIT course.
