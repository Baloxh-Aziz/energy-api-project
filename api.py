"""Energy Forecasting & Anomaly Detection API (FastAPI).

Loads the models saved by notebook 06 and serves:
  GET  /health           - is the API alive, which models are loaded
  POST /predict          - forecast one hour (1 hour ahead or 24 hours ahead)
  POST /detect-anomaly   - check whether an actual hourly value is an anomaly

Run:  uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- load the saved models once, at startup ----------
MODEL_DIR = os.environ.get("MODEL_DIR", ".")      # folder with the .pkl files from notebook 06

forecast_1h = joblib.load(os.path.join(MODEL_DIR, "energy_forecast_1h.pkl"))
forecast_24h = joblib.load(os.path.join(MODEL_DIR, "energy_forecast_24h.pkl"))
anomaly_bundle = joblib.load(os.path.join(MODEL_DIR, "anomaly_model.pkl"))

MIN_HISTORY = 168          # one week of hourly values is needed for the weekly lag

app = FastAPI(title="Energy Forecasting & Anomaly Detection API",
              description="Hourly household electricity usage (kW): forecasts and anomaly checks.")

# a ready-made example so that the /docs page is pre-filled and works with one click
EXAMPLE_HISTORY = [round(0.7 + 0.5 * np.sin(i / 24 * 2 * np.pi), 3) for i in range(MIN_HISTORY)]


# ---------- request formats ----------
class PredictRequest(BaseModel):
    history: List[float]          # hourly usage in kW, oldest first, at least 168 values
    target_time: str              # the hour to forecast, e.g. "2024-07-01 12:00:00"
    horizon: str = "1h"           # "1h" or "24h"

    model_config = {"json_schema_extra": {"examples": [
        {"history": EXAMPLE_HISTORY, "target_time": "2024-07-01 12:00:00", "horizon": "1h"}]}}


class AnomalyRequest(BaseModel):
    history: List[float]          # hourly usage up to the hour BEFORE target_time (at least 168 values)
    target_time: str              # the hour to check
    actual: float                 # measured usage (kW) in that hour
    k: Optional[float] = None     # sensitivity; lower = more alerts. Default comes from the model (5)

    model_config = {"json_schema_extra": {"examples": [
        {"history": EXAMPLE_HISTORY, "target_time": "2024-07-01 12:00:00", "actual": 4.5}]}}


# ---------- helper functions ----------
def check_inputs(history, target_time):
    # returns the target time as a Timestamp, or raises a clear error
    if len(history) < MIN_HISTORY:
        raise HTTPException(status_code=400, detail=f"history needs at least {MIN_HISTORY} hourly values")
    if not np.all(np.isfinite(history)):
        raise HTTPException(status_code=400, detail="history contains missing or invalid values")
    try:
        return pd.Timestamp(target_time)
    except Exception:
        raise HTTPException(status_code=400, detail="target_time must look like '2024-07-01 12:00:00'")


def build_features(history, target_time, steps_ahead):
    # history ends `steps_ahead` hours before target_time.
    # lag_k (usage k hours before the target) is then s[len - (k - steps_ahead + 1)].
    s = np.asarray(history, dtype=float)
    n = len(s)
    row = {"hour": target_time.hour,
           "dow": target_time.dayofweek,
           "is_weekend": int(target_time.dayofweek >= 5)}
    for k in [1, 2, 24, 48, 168]:
        if k >= steps_ahead:
            row[f"lag_{k}"] = s[n - (k - steps_ahead + 1)]
    if steps_ahead == 1:                                   # rolling features only exist for 1 hour ahead
        row["roll_mean_24"] = s[-24:].mean()
        row["roll_std_24"] = s[-24:].std(ddof=1)
        row["roll_mean_168"] = s[-168:].mean()
    return pd.DataFrame([row])


def predict_with(bundle, history, target_time, steps_ahead):
    X = build_features(history, target_time, steps_ahead)[bundle["features"]]
    prediction = bundle["model"].predict(X)[0]
    if bundle.get("base"):                                 # delta model: add the base lag back
        prediction += X[bundle["base"]].iloc[0]
    return float(prediction)


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok",
            "models": ["energy_forecast_1h", "energy_forecast_24h", "anomaly_model"],
            "min_history_hours": MIN_HISTORY}


@app.post("/predict")
def predict(req: PredictRequest):
    t = check_inputs(req.history, req.target_time)
    if req.horizon == "1h":
        value = predict_with(forecast_1h, req.history, t, steps_ahead=1)
    elif req.horizon == "24h":
        value = predict_with(forecast_24h, req.history, t, steps_ahead=24)
    else:
        raise HTTPException(status_code=400, detail="horizon must be '1h' or '24h'")
    return {"target_time": str(t), "horizon": req.horizon, "forecast_kw": round(value, 4)}


@app.post("/detect-anomaly")
def detect_anomaly(req: AnomalyRequest):
    t = check_inputs(req.history, req.target_time)
    k = req.k if req.k is not None else anomaly_bundle["k_default"]

    expected = predict_with(anomaly_bundle, req.history, t, steps_ahead=1)
    relative = (req.actual - expected) / (expected + anomaly_bundle["eps"])
    z = (relative - anomaly_bundle["median"]) / anomaly_bundle["sigma"]

    return {"target_time": str(t),
            "actual_kw": req.actual,
            "expected_kw": round(expected, 4),
            "z_score": round(float(z), 3),
            "k": k,
            "is_anomaly": bool(abs(z) > k),
            "type": "spike" if z > 0 else "drop"}
