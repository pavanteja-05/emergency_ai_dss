# ============================================================
# AI Emergency Support System — Streamlit Dashboard
# ============================================================
# Run with:  streamlit run dashboard.py
#
# Modes:
#   LIVE MODE   — polls FastAPI backend at localhost:8000
#   OFFLINE MODE — runs predictions directly (no backend needed)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
import sys
import os
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import deque

# ----------------------------------------
# API CONFIG
# Change USE_API to True to route all predictions through FastAPI.
# Make sure uvicorn is running in a separate terminal first:
#   cd backend
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ----------------------------------------
USE_API  = False          # ← flip to True to use FastAPI
API_URL  = "http://localhost:8000"

sys.path.insert(0, os.path.dirname(__file__))
from eta_adjustment import adjust_eta

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Emergency AI Dispatch",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Dark command-centre aesthetic
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');

/* ── Global ───────────────────────────────────── */
html, body, [class*="css"] {
    background-color: #050d1a;
    color: #c8d8e8;
    font-family: 'Exo 2', sans-serif;
}

/* ── Top header bar ───────────────────────────── */
.header-bar {
    background: linear-gradient(90deg, #0a1628 0%, #0d2040 50%, #0a1628 100%);
    border-bottom: 1px solid #1a3a5c;
    padding: 16px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-title {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: 1.6rem;
    color: #e8f4ff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.header-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #4a8ab5;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.pulse-dot {
    width: 10px; height: 10px;
    background: #00e5a0;
    border-radius: 50%;
    box-shadow: 0 0 8px #00e5a0;
    animation: pulse 1.4s ease-in-out infinite;
    display: inline-block;
    margin-right: 8px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

/* ── Metric cards ─────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #0d1e35 0%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #0d9ee8, transparent);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #4a8ab5;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: 2.2rem;
    color: #e8f4ff;
    line-height: 1;
}
.metric-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a8ab5;
    margin-top: 4px;
}

/* ── Risk badge colours ───────────────────────── */
.risk-low    { color: #00e5a0; }
.risk-medium { color: #f5a623; }
.risk-high   { color: #ff4d6d; }
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-low    { background: rgba(0,229,160,0.12); color: #00e5a0; border: 1px solid #00e5a040; }
.badge-medium { background: rgba(245,166,35,0.12); color: #f5a623; border: 1px solid #f5a62340; }
.badge-high   { background: rgba(255,77,109,0.12); color: #ff4d6d; border: 1px solid #ff4d6d40; }

/* ── Section headers ──────────────────────────── */
.section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4a8ab5;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ── Trip info table ──────────────────────────── */
.trip-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #0d1e35;
    font-size: 0.88rem;
}
.trip-key   { color: #4a8ab5; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; }
.trip-val   { color: #c8d8e8; font-weight: 600; }

/* ── Delay cause chip ─────────────────────────── */
.cause-chip {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 6px;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.cause-Traffic  { background: rgba(255,100,60,0.15); color: #ff6a3c; border: 1px solid #ff6a3c40; }
.cause-Weather  { background: rgba(80,160,255,0.15); color: #50a0ff; border: 1px solid #50a0ff40; }
.cause-Incident { background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid #ff4d6d40; }
.cause-NoDelay  { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid #00e5a040; }

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #080f1e;
    border-right: 1px solid #1a3a5c;
}

/* ── Streamlit overrides ──────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0d5cb5, #0a4080);
    color: #e8f4ff;
    border: 1px solid #1a6ad4;
    border-radius: 6px;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a6ad4, #0d5cb5);
    border-color: #4a9ef5;
    box-shadow: 0 0 16px rgba(74,158,245,0.3);
}
div[data-testid="stMetric"] {
    background: #0d1e35;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 12px;
}
.stSelectbox label, .stSlider label { color: #4a8ab5 !important; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADER (OFFLINE MODE)
# ============================================================

@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    try:
        return {
            "delay_cause_model":  joblib.load(os.path.join(base, "delay_cause_model.pkl")),
            "le_congestion":      joblib.load(os.path.join(base, "le_congestion.pkl")),
            "le_time_of_day":     joblib.load(os.path.join(base, "le_time_of_day.pkl")),
            "le_delay_cause":     joblib.load(os.path.join(base, "le_delay_cause.pkl")),
            "availability_model": joblib.load(os.path.join(base, "availability_model.pkl")),
            "avail_le_congestion":  joblib.load(os.path.join(base, "avail_le_congestion.pkl")),
            "avail_le_time_of_day": joblib.load(os.path.join(base, "avail_le_time_of_day.pkl")),
            "avail_le_delay_cause": joblib.load(os.path.join(base, "avail_le_delay_cause.pkl")),
        }
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Run delay_cause_model.py and availability_model.py first.")
        st.stop()


# ============================================================
# OFFLINE PREDICTION LOGIC
# ============================================================

def predict_offline(models, trip: dict) -> dict:
    """Run all three predictions locally without the FastAPI backend."""

    cong = trip["congestion_level"]
    tod  = trip["time_of_day"]
    wsev = trip["weather_severity"]
    spd  = trip["avg_speed_kmph"]
    dist = trip["distance_km"]
    exp  = trip["expected_duration_min"]
    act  = trip["active_trip_time_min"]

    # --- Delay Cause ---
    # Must match all 18 features used in delay_cause_model.py
    cong_enc = models["le_congestion"].transform([cong])[0]
    tod_enc  = models["le_time_of_day"].transform([tod])[0]

    # speed_drop_rate: not available at simulation time, use class-typical median
    # (Traffic~0.15, Weather~0.30, Incident~0.95, No Delay~0.02 → neutral default 0.15)
    sdr = trip.get("speed_drop_rate", 0.15)
    vis = trip.get("visibility_m", 800.0)

    cong_num_map = {"Low": 0, "Medium": 1, "High": 2}
    cong_num = cong_num_map.get(cong, 0)

    # Discretised bins (must match training thresholds)
    speed_bin = 0 if spd < 22 else (1 if spd < 32 else (2 if spd < 45 else 3))
    weather_bin = 0 if wsev == 0 else (1 if wsev == 1 else 2)
    sdr_bin = 0 if sdr < 0.08 else (1 if sdr < 0.35 else (2 if sdr < 0.65 else 3))

    # Physics-based interactions
    speed_x_cong    = spd * (cong_num + 1)
    speed_x_weather = spd * (wsev + 1)
    sdr_x_cong      = sdr * (cong_num + 1)
    vis_x_weather   = vis * (wsev + 1)
    sdr_x_speed     = sdr / (spd + 1)

    # Threshold flags
    peak_tods    = {"Morning Peak", "Evening Peak"}
    is_peak_hour = int(tod in peak_tods)
    is_low_speed = int(spd < 25)
    is_bad_weather = int(wsev >= 2)
    is_high_cong   = int(cong == "High")

    X_dc = pd.DataFrame([{
        "avg_speed_kmph":   spd,
        "weather_severity": wsev,
        "congestion_enc":   cong_enc,
        "tod_enc":          tod_enc,
        "visibility_m":     vis,
        "speed_drop_rate":  sdr,
        "speed_bin":        speed_bin,
        "weather_bin":      weather_bin,
        "sdr_bin":          sdr_bin,
        "speed_x_cong":     speed_x_cong,
        "speed_x_weather":  speed_x_weather,
        "sdr_x_cong":       sdr_x_cong,
        "vis_x_weather":    vis_x_weather,
        "sdr_x_speed":      sdr_x_speed,
        "is_peak_hour":     is_peak_hour,
        "is_low_speed":     is_low_speed,
        "is_bad_weather":   is_bad_weather,
        "is_high_cong":     is_high_cong,
    }])
    pred_enc    = models["delay_cause_model"].predict(X_dc)[0]
    probas      = models["delay_cause_model"].predict_proba(X_dc)[0]
    confidence  = round(float(probas.max()), 4)
    delay_cause = models["le_delay_cause"].inverse_transform([pred_enc])[0]
    classes     = models["le_delay_cause"].classes_
    class_probs = {c: round(float(p), 4) for c, p in zip(classes, probas)}

    # --- Availability ---
    av_cong = models["avail_le_congestion"].transform([cong])[0]
    av_tod  = models["avail_le_time_of_day"].transform([tod])[0]
    av_dc   = models["avail_le_delay_cause"].transform([delay_cause])[0]
    tp      = min(act / max(exp, 1), 3.0)
    risk    = wsev * (av_cong + 1)
    X_av = pd.DataFrame([{
        "expected_duration_min": exp, "distance_km": dist,
        "active_trip_time_min": act,  "time_pressure": tp,
        "avg_speed_kmph": spd, "weather_severity": wsev,
        "congestion_weather_risk": risk,
        "congestion_level_enc": av_cong, "time_of_day_enc": av_tod,
        "delay_cause_enc": av_dc
    }])
    availability = max(round(float(models["availability_model"].predict(X_av)[0]), 2), 0.0)

    # --- ETA ---
    adjusted = adjust_eta(exp, delay_cause, confidence, cong, wsev)
    added    = round(adjusted - exp, 2)
    pct      = added / max(exp, 1)
    risk_lvl = "High" if pct >= 0.30 else ("Medium" if pct >= 0.10 else "Low")

    return {
        "delay_cause":          delay_cause,
        "confidence":           confidence,
        "class_probabilities":  class_probs,
        "availability_time_min": availability,
        "base_eta_min":         exp,
        "adjusted_eta_min":     adjusted,
        "added_minutes":        added,
        "risk_level":           risk_lvl
    }


# ============================================================
# API PREDICTION LOGIC (FastAPI mode)
# ============================================================

def predict_via_api(trip: dict) -> dict:
    """
    Send trip data to the FastAPI backend (/predict/full) and return the result.
    The API server must already be running in a separate terminal:
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
    """
    payload = {
        "trip_id":               trip["trip_id"],
        "ambulance_id":          trip["ambulance_id"],
        "distance_km":           trip["distance_km"],
        "expected_duration_min": trip["expected_duration_min"],
        "active_trip_time_min":  trip["active_trip_time_min"],
        "avg_speed_kmph":        trip["avg_speed_kmph"],
        "weather_severity":      trip["weather_severity"],
        "congestion_level":      trip["congestion_level"],
        "time_of_day":           trip["time_of_day"],
        "visibility_m":          trip.get("visibility_m", 800.0),
        "speed_drop_rate":       trip.get("speed_drop_rate", 0.15),
    }
    try:
        r = requests.post(f"{API_URL}/predict/full", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ FastAPI backend is not running.\n\n"
            "Open a new terminal and run:\n"
            "```\ncd backend\nuvicorn main:app --reload --host 0.0.0.0 --port 8000\n```"
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API returned an error: {e.response.text}")
        st.stop()


def predict(models_or_none, trip: dict) -> dict:
    """
    Single entry point for predictions.
    Routes to API or offline depending on the USE_API flag at the top of this file.
    """
    if USE_API:
        return predict_via_api(trip)
    else:
        return predict_offline(models_or_none, trip)


# ============================================================
# TRIP SIMULATOR
# ============================================================

CONGESTION  = ["Low", "Medium", "High"]
WEATHER_MAP = {"Clear": 0, "Rain": 1, "Fog": 2, "Storm": 3}
TOD_OPTIONS = ["Morning Peak", "Afternoon", "Evening Peak", "Night"]
SPEED_MAP   = {"Low": (40, 60), "Medium": (25, 40), "High": (15, 25)}

# Visibility ranges per weather condition (matches delay_simulation.py)
VISIBILITY_RANGES = {"Clear": (800,1000), "Rain": (420,720), "Fog": (140,380), "Storm": (90,290)}

# Speed drop rate ranges per congestion (typical medians by class)
SDR_RANGES = {"Low": (0.00, 0.08), "Medium": (0.08, 0.35), "High": (0.35, 0.65)}

def simulate_trip() -> dict:
    cong    = random.choice(CONGESTION)
    weather = random.choice(list(WEATHER_MAP.keys()))
    tod     = random.choice(TOD_OPTIONS)
    dist    = round(random.uniform(2, 30), 2)
    speed   = round(random.uniform(*SPEED_MAP[cong]), 2)
    exp     = round((dist / 40) * 60, 2)
    active  = round(random.uniform(0, exp * 0.9), 2)
    vis_lo, vis_hi = VISIBILITY_RANGES[weather]
    visibility  = round(random.uniform(vis_lo, vis_hi), 1)
    sdr_lo, sdr_hi = SDR_RANGES[cong]
    speed_drop  = round(random.uniform(sdr_lo, sdr_hi), 4)

    return {
        "trip_id":               f"T{random.randint(1000, 9999)}",
        "ambulance_id":          f"A{random.randint(1, 50)}",
        "distance_km":           dist,
        "expected_duration_min": exp,
        "active_trip_time_min":  active,
        "avg_speed_kmph":        speed,
        "congestion_level":      cong,
        "weather_condition":     weather,
        "weather_severity":      WEATHER_MAP[weather],
        "time_of_day":           tod,
        "visibility_m":          visibility,
        "speed_drop_rate":       speed_drop
    }


# ============================================================
# CHART HELPERS
# ============================================================

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Share Tech Mono, monospace", color="#4a8ab5", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)

CAUSE_COLOURS = {
    "Traffic":  "#ff6a3c",
    "Weather":  "#50a0ff",
    "Incident": "#ff4d6d",
    "No Delay": "#00e5a0"
}

def confidence_gauge(confidence: float, delay_cause: str) -> go.Figure:
    colour = CAUSE_COLOURS.get(delay_cause, "#0d9ee8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": colour}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#1a3a5c", "tickwidth": 1},
            "bar":       {"color": colour, "thickness": 0.25},
            "bgcolor":   "#0d1e35",
            "bordercolor": "#1a3a5c",
            "steps": [
                {"range": [0,  60], "color": "#0a1020"},
                {"range": [60, 80], "color": "#0d1830"},
                {"range": [80, 100],"color": "#0d2040"},
            ],
        },
        title={"text": "MODEL CONFIDENCE", "font": {"size": 10, "color": "#4a8ab5"}}
    ))
    fig.update_layout(**CHART_THEME, height=200)
    return fig


def probability_bar(class_probs: dict) -> go.Figure:
    causes  = list(class_probs.keys())
    values  = [v * 100 for v in class_probs.values()]
    colours = [CAUSE_COLOURS.get(c, "#0d9ee8") for c in causes]

    fig = go.Figure(go.Bar(
        x=values, y=causes,
        orientation="h",
        marker=dict(color=colours, opacity=0.85),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#c8d8e8", size=10)
    ))
    fig.update_layout(
        **CHART_THEME,
        height=180,
        xaxis=dict(range=[0, 110], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#c8d8e8", size=11)),
        title=dict(text="DELAY CLASS PROBABILITIES", font=dict(size=10, color="#4a8ab5"))
    )
    return fig


def eta_comparison_bar(base: float, adjusted: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Base ETA", "Adjusted ETA"],
        y=[base, adjusted],
        marker=dict(color=["#1a5e9e", "#ff4d6d" if adjusted > base * 1.3 else "#f5a623"]),
        text=[f"{base:.1f} min", f"{adjusted:.1f} min"],
        textposition="outside",
        textfont=dict(color="#c8d8e8", size=12)
    ))
    fig.update_layout(
        **CHART_THEME,
        height=220,
        yaxis=dict(showgrid=True, gridcolor="#0d2040", zeroline=False),
        xaxis=dict(showgrid=False),
        title=dict(text="ETA COMPARISON", font=dict(size=10, color="#4a8ab5"))
    )
    return fig


def history_line(history: list, key: str, label: str, colour: str) -> go.Figure:
    vals = [h.get(key, 0) for h in history]
    fig = go.Figure(go.Scatter(
        y=vals,
        mode="lines+markers",
        line=dict(color=colour, width=2),
        marker=dict(size=4, color=colour),
        fill="tozeroy",
        fillcolor="rgba(245,166,35,0.08)" if colour == "#f5a623" else "rgba(0,229,160,0.08)" if colour == "#00e5a0" else "rgba(80,160,255,0.08)",
    ))
    fig.update_layout(
        **CHART_THEME, height=160,
        yaxis=dict(showgrid=True, gridcolor="#0d2040", zeroline=False),
        xaxis=dict(showgrid=False, showticklabels=False),
        title=dict(text=label, font=dict(size=10, color="#4a8ab5"))
    )
    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "history"       not in st.session_state: st.session_state.history       = deque(maxlen=20)
if "running"       not in st.session_state: st.session_state.running       = False
if "current_trip"  not in st.session_state: st.session_state.current_trip  = None
if "current_pred"  not in st.session_state: st.session_state.current_pred  = None
if "trip_count"    not in st.session_state: st.session_state.trip_count    = 0
if "cause_counts"  not in st.session_state: st.session_state.cause_counts  = {"Traffic":0,"Weather":0,"Incident":0,"No Delay":0}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="section-title">⚙ SYSTEM CONTROLS</div>', unsafe_allow_html=True)

    mode = st.radio("Mode", ["🔴 Live Simulation", "🔧 Manual Input"], index=0)

    st.markdown('<div class="section-title" style="margin-top:20px">⏱ SIMULATION SETTINGS</div>', unsafe_allow_html=True)
    refresh_rate = st.slider("Refresh interval (seconds)", 2, 10, 4)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ START", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏹ STOP", use_container_width=True):
            st.session_state.running = False

    if st.button("🔄 Run Once", use_container_width=True):
        trip   = simulate_trip()
        models = None if USE_API else load_models()
        pred   = predict(models, trip)
        st.session_state.current_trip = trip
        st.session_state.current_pred = pred
        st.session_state.trip_count += 1
        st.session_state.cause_counts[pred["delay_cause"]] = \
            st.session_state.cause_counts.get(pred["delay_cause"], 0) + 1
        st.session_state.history.append({
            "adjusted_eta_min":     pred["adjusted_eta_min"],
            "availability_time_min": pred["availability_time_min"],
            "confidence":           pred["confidence"],
            "delay_cause":          pred["delay_cause"]
        })

    st.markdown("---")
    st.markdown('<div class="section-title">📊 SESSION STATS</div>', unsafe_allow_html=True)
    st.markdown(f"**Trips processed:** {st.session_state.trip_count}")
    if st.session_state.history:
        avg_eta   = np.mean([h["adjusted_eta_min"]      for h in st.session_state.history])
        avg_avail = np.mean([h["availability_time_min"] for h in st.session_state.history])
        avg_conf  = np.mean([h["confidence"]            for h in st.session_state.history])
        st.markdown(f"**Avg Adj. ETA:** {avg_eta:.1f} min")
        st.markdown(f"**Avg Availability:** {avg_avail:.1f} min")
        st.markdown(f"**Avg Confidence:** {avg_conf*100:.0f}%")

    if st.button("🗑 Reset Session", use_container_width=True):
        st.session_state.history.clear()
        st.session_state.current_trip = None
        st.session_state.current_pred = None
        st.session_state.trip_count   = 0
        st.session_state.cause_counts = {"Traffic":0,"Weather":0,"Incident":0,"No Delay":0}
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#1a3a5c;text-align:center;">'
        'AI EMERGENCY DSS v1.0<br>SIMULATION MODE — NOT FOR OPERATIONAL USE'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header-bar">
    <span class="pulse-dot"></span>
    <div>
        <div class="header-title">🚑 Emergency AI Dispatch System</div>
        <div class="header-sub">AI-Based Decision Support · Simulation Mode · Real-Time Predictions</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MANUAL INPUT MODE
# ============================================================

if "Manual" in mode:
    st.markdown('<div class="section-title">🔧 MANUAL TRIP INPUT</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        trip_id      = st.text_input("Trip ID", "T9999")
        ambulance_id = st.text_input("Ambulance ID", "A01")
        distance_km  = st.number_input("Distance (km)", 2.0, 30.0, 15.0, 0.5)
    with col2:
        congestion   = st.selectbox("Congestion Level", ["Low", "Medium", "High"], index=1)
        time_of_day  = st.selectbox("Time of Day", ["Morning Peak", "Afternoon", "Evening Peak", "Night"])
        weather      = st.selectbox("Weather", ["Clear", "Rain", "Fog", "Storm"])
    with col3:
        avg_speed    = st.number_input("Avg Speed (km/h)", 5.0, 100.0, 30.0, 1.0)
        active_time  = st.number_input("Active Trip Time (min)", 0.0, 60.0, 10.0, 0.5)

    weather_sev  = WEATHER_MAP[weather]
    exp_dur      = round((distance_km / 40) * 60, 2)

    if st.button("🔮 Run Prediction", use_container_width=True):
        manual_trip = {
            "trip_id": trip_id, "ambulance_id": ambulance_id,
            "distance_km": distance_km, "expected_duration_min": exp_dur,
            "active_trip_time_min": active_time, "avg_speed_kmph": avg_speed,
            "congestion_level": congestion, "weather_condition": weather,
            "weather_severity": weather_sev, "time_of_day": time_of_day
        }
        models = None if USE_API else load_models()
        pred   = predict(models, manual_trip)
        st.session_state.current_trip = manual_trip
        st.session_state.current_pred = pred
        st.session_state.trip_count  += 1
        st.session_state.cause_counts[pred["delay_cause"]] = \
            st.session_state.cause_counts.get(pred["delay_cause"], 0) + 1

    st.markdown("---")


# ============================================================
# MAIN DASHBOARD DISPLAY
# ============================================================

trip = st.session_state.current_trip
pred = st.session_state.current_pred

if trip is None or pred is None:
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #1a3a5c;">
        <div style="font-size: 4rem; margin-bottom: 20px;">🚑</div>
        <div style="font-family: Share Tech Mono, monospace; font-size: 1rem; letter-spacing: 0.2em;">
            AWAITING DISPATCH DATA
        </div>
        <div style="font-size: 0.8rem; margin-top: 10px; color: #0d2040;">
            Press ▶ START or 🔄 Run Once in the sidebar to begin simulation
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Row 1: Key Metrics ──────────────────────────────────────

    cause_css = pred["delay_cause"].replace(" ", "")
    risk      = pred["risk_level"]
    risk_cls  = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(risk, "badge-low")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)

    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚑 Ambulance</div>
            <div class="metric-value" style="font-size:1.5rem;">{trip['ambulance_id']}</div>
            <div class="metric-unit">Trip {trip['trip_id']}</div>
        </div>""", unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📍 Distance</div>
            <div class="metric-value">{trip['distance_km']}</div>
            <div class="metric-unit">km</div>
        </div>""", unsafe_allow_html=True)

    with mc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱ Adjusted ETA</div>
            <div class="metric-value" style="color:#f5a623;">{pred['adjusted_eta_min']:.1f}</div>
            <div class="metric-unit">minutes (+{pred['added_minutes']:.1f} added)</div>
        </div>""", unsafe_allow_html=True)

    with mc4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🕐 Availability</div>
            <div class="metric-value" style="color:#00e5a0;">{pred['availability_time_min']:.1f}</div>
            <div class="metric-unit">minutes until free</div>
        </div>""", unsafe_allow_html=True)

    with mc5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠ Risk Level</div>
            <div class="metric-value" style="font-size:1.6rem; margin:6px 0;">
                <span class="badge {risk_cls}">{risk}</span>
            </div>
            <div class="metric-unit">ETA risk assessment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Left (Trip Info + Delay Cause) | Right (Charts) ──

    left, right = st.columns([1, 2])

    with left:
        # Trip Details
        st.markdown('<div class="section-title">📋 TRIP DETAILS</div>', unsafe_allow_html=True)
        rows = [
            ("CONGESTION",   trip["congestion_level"]),
            ("WEATHER",      trip["weather_condition"]),
            ("TIME OF DAY",  trip["time_of_day"]),
            ("AVG SPEED",    f"{trip['avg_speed_kmph']} km/h"),
            ("ACTIVE TIME",  f"{trip['active_trip_time_min']:.1f} min"),
            ("EXPECTED DUR", f"{trip['expected_duration_min']:.1f} min"),
        ]
        html_rows = "".join([
            f'<div class="trip-row"><span class="trip-key">{k}</span><span class="trip-val">{v}</span></div>'
            for k, v in rows
        ])
        st.markdown(f'<div style="background:#080f1e;border:1px solid #1a3a5c;border-radius:8px;padding:12px;">{html_rows}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Delay Cause
        st.markdown('<div class="section-title">🔍 DELAY CAUSE PREDICTION</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center; padding: 16px 0;">'
            f'<span class="cause-chip cause-{cause_css}">{pred["delay_cause"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(confidence_gauge(pred["confidence"], pred["delay_cause"]),
                        use_container_width=True, config={"displayModeBar": False})

    with right:
        # Class Probabilities
        st.plotly_chart(probability_bar(pred["class_probabilities"]),
                        use_container_width=True, config={"displayModeBar": False})

        # ETA Comparison
        st.plotly_chart(eta_comparison_bar(pred["base_eta_min"], pred["adjusted_eta_min"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Row 3: History Trends ───────────────────────────────────

    if len(st.session_state.history) > 1:
        st.markdown('<div class="section-title">📈 SESSION TRENDS</div>', unsafe_allow_html=True)

        h1, h2, h3 = st.columns(3)
        history = list(st.session_state.history)

        with h1:
            st.plotly_chart(
                history_line(history, "adjusted_eta_min", "ADJUSTED ETA (min)", "#f5a623"),
                use_container_width=True, config={"displayModeBar": False}
            )
        with h2:
            st.plotly_chart(
                history_line(history, "availability_time_min", "AVAILABILITY (min)", "#00e5a0"),
                use_container_width=True, config={"displayModeBar": False}
            )
        with h3:
            st.plotly_chart(
                history_line(history, "confidence", "CONFIDENCE", "#50a0ff"),
                use_container_width=True, config={"displayModeBar": False}
            )

        # Cause distribution donut
        st.markdown('<div class="section-title">🍩 DELAY CAUSE DISTRIBUTION (SESSION)</div>', unsafe_allow_html=True)

        cause_hist  = [h["delay_cause"] for h in history]
        cause_df    = pd.Series(cause_hist).value_counts().reset_index()
        cause_df.columns = ["cause", "count"]

        fig_donut = go.Figure(go.Pie(
            labels=cause_df["cause"],
            values=cause_df["count"],
            hole=0.55,
            marker=dict(colors=[CAUSE_COLOURS.get(c, "#888") for c in cause_df["cause"]]),
            textfont=dict(color="#c8d8e8"),
            hovertemplate="%{label}: %{value} trips (%{percent})<extra></extra>"
        ))
        fig_donut.update_layout(**CHART_THEME, height=240,
                                legend=dict(font=dict(color="#c8d8e8"), orientation="h"))
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})


# ============================================================
# TIMESTAMP FOOTER
# ============================================================

st.markdown(
    f'<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#1a3a5c;'
    f'text-align:right;margin-top:20px;">LAST UPDATED: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</div>',
    unsafe_allow_html=True
)


# ============================================================
# AUTO-REFRESH LOOP (LIVE MODE)
# ============================================================

if "Live" in mode and st.session_state.running:
    time.sleep(refresh_rate)
    trip   = simulate_trip()
    models = None if USE_API else load_models()
    pred   = predict(models, trip)

    st.session_state.current_trip = trip
    st.session_state.current_pred = pred
    st.session_state.trip_count  += 1
    st.session_state.cause_counts[pred["delay_cause"]] = \
        st.session_state.cause_counts.get(pred["delay_cause"], 0) + 1
    st.session_state.history.append({
        "adjusted_eta_min":      pred["adjusted_eta_min"],
        "availability_time_min": pred["availability_time_min"],
        "confidence":            pred["confidence"],
        "delay_cause":           pred["delay_cause"]
    })

    st.rerun()
