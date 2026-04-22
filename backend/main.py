# ============================================================
# AI-Based Emergency Support System — FastAPI Backend
# ============================================================
# Endpoints:
#   POST /predict/delay-cause       → Classify delay cause
#   POST /predict/availability      → Predict ambulance availability
#   POST /predict/eta               → Full ETA adjustment
#   POST /predict/full              → All three in one call
#   GET  /health                    → Health check
#   GET  /simulate/live-trip        → Simulated live trip data
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import random
from datetime import datetime

from models import (
    DelayCauseRequest,
    DelayCauseResponse,
    AvailabilityRequest,
    AvailabilityResponse,
    ETARequest,
    ETAResponse,
    FullPredictionRequest,
    FullPredictionResponse,
    LiveTripResponse
)
from eta_adjustment import adjust_eta


# ----------------------------------------
# STARTUP: LOAD ALL MODELS & ENCODERS
# ----------------------------------------

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, release on shutdown."""
    print("[STARTUP] Loading models and encoders...")

    try:
        # Delay cause classifier
        ml_models["delay_cause_model"]   = joblib.load("delay_cause_model.pkl")
        ml_models["le_congestion"]        = joblib.load("le_congestion.pkl")
        ml_models["le_time_of_day"]       = joblib.load("le_time_of_day.pkl")
        ml_models["le_delay_cause"]       = joblib.load("le_delay_cause.pkl")

        # Availability predictor
        ml_models["availability_model"]        = joblib.load("availability_model.pkl")
        ml_models["avail_le_congestion"]       = joblib.load("avail_le_congestion.pkl")
        ml_models["avail_le_time_of_day"]      = joblib.load("avail_le_time_of_day.pkl")
        ml_models["avail_le_delay_cause"]      = joblib.load("avail_le_delay_cause.pkl")

        print("[STARTUP] All models loaded successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Model file not found: {e}")
        print("Run delay_cause_model.py and availability_model.py first to generate .pkl files.")
        raise

    yield  # App runs here

    print("[SHUTDOWN] Cleaning up models...")
    ml_models.clear()


# ----------------------------------------
# APP INITIALISATION
# ----------------------------------------

app = FastAPI(
    title="AI Emergency Support System API",
    description=(
        "Simulation-driven Decision Support System for emergency ambulance dispatch. "
        "Provides delay cause classification, ambulance availability prediction, "
        "and risk-adjusted ETA estimates."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Allow all origins for dashboard/frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def encode_delay_cause_features(data: DelayCauseRequest) -> np.ndarray:
    """
    Encode and prepare all 18 features for the delay cause classifier.
    Must exactly match the feature engineering in delay_cause_model.py.
    """
    s    = data.avg_speed_kmph
    w    = data.weather_severity
    c    = data.congestion_level
    t    = data.time_of_day
    vis  = data.visibility_m
    sdr  = data.speed_drop_rate

    cong_enc = ml_models["le_congestion"].transform([c])[0]
    tod_enc  = ml_models["le_time_of_day"].transform([t])[0]

    # Numeric congestion (matches delay_cause_model.py mapping)
    cong_num = {"Low": 0, "Medium": 1, "High": 2}[c]

    # Discretised bins
    speed_bin   = 0 if s < 22  else (1 if s < 32  else (2 if s < 45 else 3))
    weather_bin = 0 if w == 0  else (1 if w == 1  else 2)
    sdr_bin     = 0 if sdr < 0.08 else (1 if sdr < 0.35 else (2 if sdr < 0.65 else 3))

    # Physics-based interactions
    speed_x_cong    = s   * (cong_num + 1)
    speed_x_weather = s   * (w + 1)
    sdr_x_cong      = sdr * (cong_num + 1)
    vis_x_weather   = vis * (w + 1)
    sdr_x_speed     = sdr / (s + 1)

    # Threshold flags
    peak_set     = {"Morning Peak", "Evening Peak"}
    is_peak_hour = int(t in peak_set)
    is_low_speed = int(s < 25)
    is_bad_weather = int(w >= 2)
    is_high_cong = int(c == "High")

    return np.array([[
        s, w, cong_enc, tod_enc, vis, sdr,
        speed_bin, weather_bin, sdr_bin,
        speed_x_cong, speed_x_weather, sdr_x_cong, vis_x_weather, sdr_x_speed,
        is_peak_hour, is_low_speed, is_bad_weather, is_high_cong
    ]])


def encode_availability_features(data: AvailabilityRequest) -> pd.DataFrame:
    """Encode and prepare features for the availability predictor."""
    cong_enc = ml_models["avail_le_congestion"].transform([data.congestion_level])[0]
    tod_enc  = ml_models["avail_le_time_of_day"].transform([data.time_of_day])[0]
    dc_enc   = ml_models["avail_le_delay_cause"].transform([data.delay_cause])[0]

    time_pressure          = min(data.active_trip_time_min / max(data.expected_duration_min, 1), 3.0)
    congestion_weather_risk = data.weather_severity * (cong_enc + 1)

    return pd.DataFrame([{
        "expected_duration_min":  data.expected_duration_min,
        "distance_km":            data.distance_km,
        "active_trip_time_min":   data.active_trip_time_min,
        "time_pressure":          time_pressure,
        "avg_speed_kmph":         data.avg_speed_kmph,
        "weather_severity":       data.weather_severity,
        "congestion_weather_risk": congestion_weather_risk,
        "congestion_level_enc":   cong_enc,
        "time_of_day_enc":        tod_enc,
        "delay_cause_enc":        dc_enc
    }])


# ============================================================
# ENDPOINTS
# ============================================================

# ----------------------------------------
# GET /health
# ----------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Returns system status and loaded model names."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": list(ml_models.keys()),
        "version": "1.0.0"
    }


# ----------------------------------------
# POST /predict/delay-cause
# ----------------------------------------

@app.post("/predict/delay-cause", response_model=DelayCauseResponse, tags=["Predictions"])
def predict_delay_cause(data: DelayCauseRequest):
    """
    Classify the most likely cause of delay for an ambulance trip.

    Returns predicted delay cause and model confidence score.
    """
    try:
        X = encode_delay_cause_features(data)

        model       = ml_models["delay_cause_model"]
        prediction  = model.predict(X)[0]
        probas      = model.predict_proba(X)[0]
        confidence  = round(float(probas.max()), 4)

        delay_cause = ml_models["le_delay_cause"].inverse_transform([prediction])[0]

        # Build class probability breakdown
        classes      = ml_models["le_delay_cause"].classes_
        class_probas = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}

        return DelayCauseResponse(
            delay_cause=delay_cause,
            confidence=confidence,
            class_probabilities=class_probas
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Encoding error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ----------------------------------------
# POST /predict/availability
# ----------------------------------------

@app.post("/predict/availability", response_model=AvailabilityResponse, tags=["Predictions"])
def predict_availability(data: AvailabilityRequest):
    """
    Predict how many minutes until the ambulance becomes available.

    Uses trip progress, conditions, and delay cause to estimate remaining time.
    """
    try:
        X = encode_availability_features(data)

        model     = ml_models["availability_model"]
        predicted = float(model.predict(X)[0])
        predicted = max(round(predicted, 2), 0.0)

        # Estimated clock time of availability
        now              = datetime.utcnow()
        available_at_utc = now.isoformat()  # placeholder; dashboard applies real offset

        return AvailabilityResponse(
            availability_time_min=predicted,
            available_at_utc=available_at_utc
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Encoding error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ----------------------------------------
# POST /predict/eta
# ----------------------------------------

@app.post("/predict/eta", response_model=ETAResponse, tags=["Predictions"])
def predict_eta(data: ETARequest):
    """
    Adjust the base ETA using predicted delay cause, congestion, and weather severity.

    Returns both the base ETA and the risk-adjusted ETA.
    """
    adjusted = adjust_eta(
        base_eta        = data.base_eta_min,
        predicted_delay = data.predicted_delay_cause,
        confidence      = data.confidence,
        congestion      = data.congestion_level,
        weather_severity= data.weather_severity
    )

    added_minutes = round(adjusted - data.base_eta_min, 2)

    return ETAResponse(
        base_eta_min    = data.base_eta_min,
        adjusted_eta_min= adjusted,
        added_minutes   = added_minutes,
        risk_level      = _classify_risk(added_minutes, data.base_eta_min)
    )


def _classify_risk(added: float, base: float) -> str:
    """Classify ETA risk level based on percentage increase."""
    if base == 0:
        return "Unknown"
    pct = added / base
    if pct < 0.10:
        return "Low"
    elif pct < 0.30:
        return "Medium"
    else:
        return "High"


# ----------------------------------------
# POST /predict/full
# ----------------------------------------

@app.post("/predict/full", response_model=FullPredictionResponse, tags=["Predictions"])
def predict_full(data: FullPredictionRequest):
    """
    Run all three predictions in a single API call:
      1. Delay cause classification
      2. Ambulance availability
      3. Risk-adjusted ETA

    Designed for the real-time dashboard to poll this endpoint.
    """
    # --- Step 1: Delay Cause ---
    delay_req = DelayCauseRequest(
        avg_speed_kmph   = data.avg_speed_kmph,
        weather_severity = data.weather_severity,
        congestion_level = data.congestion_level,
        time_of_day      = data.time_of_day,
        visibility_m     = data.visibility_m,
        speed_drop_rate  = data.speed_drop_rate
    )
    delay_resp = predict_delay_cause(delay_req)

    # --- Step 2: Availability ---
    avail_req = AvailabilityRequest(
        expected_duration_min = data.expected_duration_min,
        distance_km           = data.distance_km,
        active_trip_time_min  = data.active_trip_time_min,
        avg_speed_kmph        = data.avg_speed_kmph,
        weather_severity      = data.weather_severity,
        congestion_level      = data.congestion_level,
        time_of_day           = data.time_of_day,
        delay_cause           = delay_resp.delay_cause
    )
    avail_resp = predict_availability(avail_req)

    # --- Step 3: ETA Adjustment ---
    eta_req = ETARequest(
        base_eta_min          = data.expected_duration_min,
        predicted_delay_cause = delay_resp.delay_cause,
        confidence            = delay_resp.confidence,
        congestion_level      = data.congestion_level,
        weather_severity      = data.weather_severity
    )
    eta_resp = predict_eta(eta_req)

    return FullPredictionResponse(
        trip_id              = data.trip_id,
        ambulance_id         = data.ambulance_id,
        timestamp            = datetime.utcnow().isoformat(),
        delay_cause          = delay_resp.delay_cause,
        confidence           = delay_resp.confidence,
        class_probabilities  = delay_resp.class_probabilities,
        availability_time_min= avail_resp.availability_time_min,
        base_eta_min         = eta_resp.base_eta_min,
        adjusted_eta_min     = eta_resp.adjusted_eta_min,
        added_minutes        = eta_resp.added_minutes,
        risk_level           = eta_resp.risk_level
    )


# ----------------------------------------
# GET /simulate/live-trip
# ----------------------------------------

CONGESTION_LEVELS = ["Low", "Medium", "High"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Fog", "Storm"]
WEATHER_SEVERITY_MAP = {"Clear": 0, "Rain": 1, "Fog": 2, "Storm": 3}
TIME_OF_DAY_OPTIONS = ["Morning Peak", "Afternoon", "Evening Peak", "Night"]
SPEED_MAP = {"Low": (40, 60), "Medium": (25, 40), "High": (15, 25)}

@app.get("/simulate/live-trip", response_model=LiveTripResponse, tags=["Simulation"])
def simulate_live_trip():
    """
    Generate a simulated live ambulance trip with randomised realistic values.

    Used by the Streamlit dashboard to demo real-time AI predictions
    without needing real ambulance data.
    """
    trip_id      = f"T{random.randint(1000, 9999)}"
    ambulance_id = f"A{random.randint(1, 50)}"
    distance_km  = round(random.uniform(2, 30), 2)

    congestion   = random.choice(CONGESTION_LEVELS)
    weather      = random.choice(WEATHER_CONDITIONS)
    time_of_day  = random.choice(TIME_OF_DAY_OPTIONS)

    speed_range  = SPEED_MAP[congestion]
    avg_speed    = round(random.uniform(*speed_range), 2)
    weather_sev  = WEATHER_SEVERITY_MAP[weather]

    expected_dur = round((distance_km / 40) * 60, 2)   # at ideal 40 km/h
    active_time  = round(random.uniform(0, expected_dur * 0.9), 2)

    return LiveTripResponse(
        trip_id              = trip_id,
        ambulance_id         = ambulance_id,
        distance_km          = distance_km,
        expected_duration_min= expected_dur,
        active_trip_time_min = active_time,
        avg_speed_kmph       = avg_speed,
        congestion_level     = congestion,
        weather_condition    = weather,
        weather_severity     = weather_sev,
        time_of_day          = time_of_day,
        timestamp            = datetime.utcnow().isoformat()
    )
