import pandas as pd
import numpy as np

# -------------------------------
# STEP 1: LOAD EXISTING TRIP DATA
# -------------------------------

trip_df = pd.read_csv("trip_time_data.csv", parse_dates=["start_time"])

# -------------------------------
# STEP 2: EXTRACT HOUR FROM TIME
# -------------------------------

trip_df["hour"] = trip_df["start_time"].dt.hour

# -------------------------------
# STEP 3: DEFINE TIME OF DAY
# -------------------------------

def get_time_of_day(hour):
    if 6 <= hour < 10:
        return "Morning Peak"
    elif 10 <= hour < 16:
        return "Afternoon"
    elif 16 <= hour < 21:
        return "Evening Peak"
    else:
        return "Night"

trip_df["time_of_day"] = trip_df["hour"].apply(get_time_of_day)

# -------------------------------
# STEP 4: ASSIGN CONGESTION LEVEL
# -------------------------------

def get_congestion(time_of_day):
    if time_of_day in ["Morning Peak", "Evening Peak"]:
        return np.random.choice(["High", "Medium"], p=[0.7, 0.3])
    elif time_of_day == "Afternoon":
        return np.random.choice(["Medium", "Low"], p=[0.6, 0.4])
    else:  # Night
        return np.random.choice(["Low", "Medium"], p=[0.8, 0.2])

trip_df["congestion_level"] = trip_df["time_of_day"].apply(get_congestion)

# -------------------------------
# STEP 5: ASSIGN AVERAGE SPEED
# -------------------------------

def get_speed(congestion):
    if congestion == "High":
        return np.random.uniform(15, 25)
    elif congestion == "Medium":
        return np.random.uniform(25, 40)
    else:  # Low
        return np.random.uniform(40, 60)

trip_df["avg_speed_kmph"] = trip_df["congestion_level"].apply(get_speed).round(2)

# -------------------------------
# STEP 6: CLEAN UP
# -------------------------------

trip_df.drop(columns=["hour"], inplace=True)

# -------------------------------
# STEP 7: VIEW SAMPLE OUTPUT
# -------------------------------

print("Sample data after Traffic Simulation:")
print(trip_df.head())

# -------------------------------
# STEP 8: SAVE UPDATED DATASET
# -------------------------------

trip_df.to_csv("trip_time_traffic_data.csv", index=False)

print("\nTraffic simulation completed.")
print("Updated dataset saved as 'trip_time_traffic_data.csv'")
