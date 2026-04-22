import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------------
# STEP 1: CONFIGURATION
# -------------------------------

NUM_TRIPS = 5000          # Total number of ambulance trips
NUM_AMBULANCES = 50       # Total ambulances in the system
IDEAL_SPEED_KMPH = 40     # Ideal average speed (km/h)

# -------------------------------
# STEP 2: GENERATE TRIP IDS
# -------------------------------

trip_ids = [f"T{i+1}" for i in range(NUM_TRIPS)]

# -------------------------------
# STEP 3: GENERATE AMBULANCE IDS
# -------------------------------

ambulance_ids = [f"A{i+1}" for i in range(NUM_AMBULANCES)]

# Assign one ambulance to each trip
assigned_ambulances = np.random.choice(ambulance_ids, NUM_TRIPS)

# -------------------------------
# STEP 4: SIMULATE START TIMES
# -------------------------------

start_date = datetime(2025, 9, 1)

start_times = [
    start_date + timedelta(
        days=np.random.randint(0, 30),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    )
    for _ in range(NUM_TRIPS)
]

# -------------------------------
# STEP 5: SIMULATE DISTANCES
# -------------------------------

# Distance in km (realistic city range)
distances_km = np.random.uniform(2, 30, NUM_TRIPS)

# -------------------------------
# STEP 6: CALCULATE EXPECTED DURATION
# -------------------------------

# Expected duration in minutes (no delays)
expected_duration_min = (distances_km / IDEAL_SPEED_KMPH) * 60

# -------------------------------
# STEP 7: CREATE DATAFRAME
# -------------------------------

trip_df = pd.DataFrame({
    "trip_id": trip_ids,
    "ambulance_id": assigned_ambulances,
    "start_time": start_times,
    "distance_km": distances_km.round(2),
    "expected_duration_min": expected_duration_min.round(2)
})

# -------------------------------
# STEP 8: VIEW SAMPLE OUTPUT
# -------------------------------

print("Sample Trip & Time Simulation Data:")
print(trip_df.head())

# -------------------------------
# STEP 9: SAVE DATASET
# -------------------------------

trip_df.to_csv("trip_time_data.csv", index=False)

print("\nDataset saved as 'trip_time_data.csv'")
print(f"Total trips generated: {len(trip_df)}")
