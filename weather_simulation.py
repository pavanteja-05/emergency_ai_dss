import pandas as pd
import numpy as np

# -------------------------------
# STEP 1: LOAD TRAFFIC DATA
# -------------------------------

df = pd.read_csv("trip_time_traffic_data.csv", parse_dates=["start_time"])

# -------------------------------
# STEP 2: DEFINE WEATHER TYPES
# -------------------------------

weather_conditions = ["Clear", "Rain", "Fog", "Storm"]
weather_probabilities = [0.55, 0.25, 0.15, 0.05]

df["weather_condition"] = np.random.choice(
    weather_conditions,
    size=len(df),
    p=weather_probabilities
)

# -------------------------------
# STEP 3: ASSIGN WEATHER SEVERITY
# -------------------------------

weather_severity_map = {
    "Clear": 0,
    "Rain": 1,
    "Fog": 2,
    "Storm": 3
}

df["weather_severity"] = df["weather_condition"].map(weather_severity_map)

# -------------------------------
# STEP 4: ASSIGN VISIBILITY
# -------------------------------

def get_visibility(weather):
    if weather == "Clear":
        return np.random.uniform(800, 1000)
    elif weather == "Rain":
        return np.random.uniform(400, 700)
    elif weather == "Fog":
        return np.random.uniform(100, 300)
    else:  # Storm
        return np.random.uniform(200, 400)

df["visibility_m"] = df["weather_condition"].apply(get_visibility).round(0)

# -------------------------------
# STEP 5: VIEW SAMPLE OUTPUT
# -------------------------------

print("Sample data after Weather Simulation:")
print(df.head())

# -------------------------------
# STEP 6: SAVE UPDATED DATASET
# -------------------------------

df.to_csv("trip_time_traffic_weather_data.csv", index=False)

print("\nWeather simulation completed.")
print("Updated dataset saved as 'trip_time_traffic_weather_data.csv'")
