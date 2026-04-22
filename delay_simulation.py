"""
DELAY SIMULATION — REALISTIC NOISE VERSION
===========================================
Changes from the previous "too clean" version:

1. WIDER SPEED OVERLAPS
   - Traffic / Incident overlap zone widened: 18-28 km/h (was 22-24)
   - Weather bleeds into No Delay range at high end
   - No hard clips — tails allowed to cross class boundaries

2. SENSOR NOISE
   - avg_speed_kmph  : ± Gaussian noise (std=3.5 km/h) simulating GPS jitter
   - visibility_m    : ± 10% uniform noise simulating sensor drift
   - speed_drop_rate : ± Gaussian noise (std=0.08) simulating telemetry lag

3. CO-OCCURRING CONDITIONS
   - Storm during Evening Peak: speed and sdr deliberately ambiguous
     between Weather and Incident
   - High congestion + moderate weather: Traffic/Weather boundary blurred

4. LABEL NOISE (~4%)
   - ~4% of rows get their delay_cause flipped to a plausible neighbour
     (Traffic->Incident, Weather->No Delay) simulating dispatcher
     mislabelling, common in real EMS logs

5. AVAILABILITY TARGET NOISE
   - availability_time_min gets +-15% noise so R2 drops from 0.97 to ~0.93

Expected downstream model performance:
   Delay cause classifier : 90-94% accuracy, F1 per class 0.88-0.93
   Availability predictor : R2 0.92-0.95, MAE ~1.8-2.5 min
   Confidence scores      : 0.60-0.92 average (no more 1.0 everywhere)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(2024)

N_PER_CLASS  = 1250
N_AMBULANCES = 50
IDEAL_SPEED  = 40.0
START_DATE   = datetime(2025, 9, 1)

WEATHER_NAME = {0: "Clear", 1: "Rain", 2: "Fog", 3: "Storm"}

def add_speed_noise(speed):
    return float(np.clip(speed + np.random.normal(0, 3.5), 5, 70))

def add_sdr_noise(sdr):
    return float(np.clip(sdr + np.random.normal(0, 0.08), 0, 2.5))

def add_visibility_noise(vis):
    factor = np.random.uniform(0.90, 1.10)
    return float(round(np.clip(vis * factor, 50, 1100)))

def add_availability_noise(avail):
    factor = np.random.uniform(0.90, 1.10)
    return float(round(max(avail * factor, 1.0), 2))

def visibility_from_severity(wsev):
    ranges = {0: (800,1000), 1: (420,720), 2: (140,380), 3: (90,290)}
    lo, hi = ranges[int(wsev)]
    return float(round(np.random.uniform(lo, hi)))

def build_row(idx, speed, wsev, cong, tod, delay_frac, cause, speed_drop_rate):
    dist   = round(np.random.uniform(2, 30), 2)
    exp    = round((dist / IDEAL_SPEED) * 60, 2)
    delay  = round(delay_frac * exp, 2)
    actual = round(max(exp + delay, exp * 0.90), 2)
    af     = np.random.uniform(0.0, 0.9)
    active = round(actual * af, 2)
    avail_clean = actual * (1 - af) + np.random.uniform(2, 8)
    avail  = add_availability_noise(round(avail_clean, 2))
    noisy_speed = add_speed_noise(speed)
    noisy_sdr   = add_sdr_noise(speed_drop_rate)
    noisy_vis   = add_visibility_noise(visibility_from_severity(wsev))
    return dict(
        trip_id=f"T{idx}",
        ambulance_id=f"A{np.random.randint(1, N_AMBULANCES+1)}",
        start_time=START_DATE + timedelta(
            days=int(np.random.randint(0, 30)),
            hours=int(np.random.randint(0, 24)),
            minutes=int(np.random.randint(0, 60))
        ),
        distance_km=dist,
        expected_duration_min=exp,
        time_of_day=tod,
        congestion_level=cong,
        avg_speed_kmph=round(noisy_speed, 2),
        weather_condition=WEATHER_NAME[int(wsev)],
        weather_severity=int(wsev),
        visibility_m=noisy_vis,
        speed_drop_rate=round(noisy_sdr, 4),
        delay_minutes=delay,
        delay_cause=cause,
        actual_duration_min=actual,
        active_trip_time_min=active,
        availability_time_min=avail,
    )

def tod_from_probs(choices, probs):
    return np.random.choice(choices, p=probs)

def gen_no_delay(n):
    rows = []
    for i in range(n):
        tod   = tod_from_probs(["Night","Afternoon","Morning Peak","Evening Peak"],[0.50,0.35,0.08,0.07])
        wsev  = 0
        speed = np.clip(np.random.normal(46, 7), 33, 65)
        cong  = np.random.choice(["Low","Medium"], p=[0.88,0.12])
        sdr   = np.random.uniform(0.00, 0.08)
        rows.append(build_row(i, speed, wsev, cong, tod, np.random.uniform(-0.04,0.06), "No Delay", sdr))
    return rows

def gen_traffic(n):
    rows = []
    for i in range(n):
        tod   = tod_from_probs(["Morning Peak","Evening Peak","Afternoon","Night"],[0.42,0.38,0.14,0.06])
        wsev  = int(np.random.choice([0,1], p=[0.72,0.28]))
        speed = np.clip(np.random.normal(28, 6), 18, 42)
        cong  = np.random.choice(["High","Medium"], p=[0.65,0.35])
        sdr   = np.random.uniform(0.05, 0.38)
        if cong == "High" and wsev == 1 and np.random.random() < 0.15:
            wsev = 2
        rows.append(build_row(i, speed, wsev, cong, tod, np.random.uniform(0.18,0.52), "Traffic", sdr))
    return rows

def gen_weather(n):
    rows = []
    for i in range(n):
        tod   = tod_from_probs(["Night","Afternoon","Morning Peak","Evening Peak"],[0.30,0.30,0.20,0.20])
        wsev  = int(np.random.choice([2,3,1], p=[0.48,0.35,0.17]))
        speed = np.clip(np.random.normal(35, 9), 22, 55)
        cong  = np.random.choice(["Low","Medium","High"], p=[0.44,0.40,0.16])
        sdr   = np.random.uniform(0.08, 0.55)
        if wsev == 3 and tod == "Evening Peak" and np.random.random() < 0.20:
            speed = np.clip(speed * 0.65, 10, 30)
            sdr   = np.random.uniform(0.50, 0.90)
        rows.append(build_row(i, speed, wsev, cong, tod, np.random.uniform(0.10,0.45), "Weather", sdr))
    return rows

def gen_incident(n):
    rows = []
    for i in range(n):
        tod   = tod_from_probs(["Morning Peak","Evening Peak","Afternoon","Night"],[0.28,0.30,0.24,0.18])
        wsev  = int(np.random.choice([1,0,2,3], p=[0.35,0.28,0.27,0.10]))
        speed = np.clip(np.random.normal(17, 5), 6, 28)
        cong  = np.random.choice(["High","Medium"], p=[0.80,0.20])
        sdr   = np.random.uniform(0.45, 1.35)
        rows.append(build_row(i, speed, wsev, cong, tod, np.random.uniform(0.28,0.65), "Incident", sdr))
    return rows

NEIGHBOUR = {
    "Traffic":  "Incident",
    "Incident": "Traffic",
    "Weather":  "No Delay",
    "No Delay": "Weather",
}

def apply_label_noise(df, noise_rate=0.04):
    n_flip   = int(len(df) * noise_rate)
    flip_idx = np.random.choice(df.index, size=n_flip, replace=False)
    original = df.loc[flip_idx, "delay_cause"].copy()
    df.loc[flip_idx, "delay_cause"] = original.map(NEIGHBOUR)
    flipped  = (df.loc[flip_idx, "delay_cause"] != original).sum()
    print(f"  Label noise applied: {flipped} labels flipped ({noise_rate*100:.0f}% rate)")
    return df

print("Generating realistic noisy dataset ...")

all_rows = (
    gen_no_delay(N_PER_CLASS) +
    gen_traffic(N_PER_CLASS) +
    gen_weather(N_PER_CLASS) +
    gen_incident(N_PER_CLASS)
)

df = pd.DataFrame(all_rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["trip_id"] = [f"T{i+1}" for i in range(len(df))]

print(f"\nTotal rows : {len(df)}")
print("\nClass distribution (before label noise):")
print(df["delay_cause"].value_counts())

print("\nApplying label noise...")
df = apply_label_noise(df, noise_rate=0.04)

print("\nClass distribution (after label noise):")
print(df["delay_cause"].value_counts())

print("\nSpeed stats by class:")
print(df.groupby("delay_cause")["avg_speed_kmph"].describe()[["mean","std","min","25%","75%","max"]].round(2))

print("\nSpeed drop rate stats by class:")
print(df.groupby("delay_cause")["speed_drop_rate"].describe()[["mean","std","min","max"]].round(3))

t_in_inc = (df[df["delay_cause"]=="Traffic"]["avg_speed_kmph"] < 28).mean()
i_in_tra = (df[df["delay_cause"]=="Incident"]["avg_speed_kmph"] > 18).mean()
w_in_nd  = (df[df["delay_cause"]=="Weather"]["avg_speed_kmph"] > 42).mean()
print(f"\nOverlap diagnostics (target: ~10-18%):")
print(f"  Traffic rows with speed < 28  : {t_in_inc:.1%}")
print(f"  Incident rows with speed > 18 : {i_in_tra:.1%}")
print(f"  Weather rows with speed > 42  : {w_in_nd:.1%}")

df.to_csv("final_simulated_emergency_data.csv", index=False)
print("\nSaved: final_simulated_emergency_data.csv")
