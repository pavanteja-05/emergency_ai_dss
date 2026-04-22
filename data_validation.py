import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: LOAD FINAL DATASET
# -------------------------------

df = pd.read_csv(
    "final_simulated_emergency_data.csv",
    parse_dates=["start_time"]
)

print("\nDataset loaded successfully")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

# -------------------------------
# STEP 2: BASIC SANITY STATISTICS
# -------------------------------

print("\nBasic Statistics (Key Numerical Columns):")
print(df[
    [
        "distance_km",
        "expected_duration_min",
        "actual_duration_min",
        "delay_minutes",
        "availability_time_min"
    ]
].describe())

# -------------------------------
# STEP 3: DELAY CAUSE DISTRIBUTION
# -------------------------------

print("\nDelay Cause Distribution:")
print(df["delay_cause"].value_counts())

df["delay_cause"].value_counts().plot(
    kind="bar",
    title="Delay Cause Distribution"
)

plt.xlabel("Delay Cause")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig("plot_delay_cause_distribution.png", dpi=150)
plt.clf()
print("Saved: plot_delay_cause_distribution.png")

# -------------------------------
# STEP 4: SPEED VS CONGESTION CHECK
# -------------------------------

df.boxplot(
    column="avg_speed_kmph",
    by="congestion_level"
)

plt.title("Average Speed vs Congestion Level")
plt.suptitle("")
plt.xlabel("Congestion Level")
plt.ylabel("Average Speed (km/h)")
plt.tight_layout()
plt.savefig("plot_speed_vs_congestion.png", dpi=150)
plt.clf()
print("Saved: plot_speed_vs_congestion.png")

# -------------------------------
# STEP 5: WEATHER IMPACT ON DELAY
# -------------------------------

df.boxplot(
    column="delay_minutes",
    by="weather_condition"
)

plt.title("Delay Minutes by Weather Condition")
plt.suptitle("")
plt.xlabel("Weather Condition")
plt.ylabel("Delay (minutes)")
plt.tight_layout()
plt.savefig("plot_delay_by_weather.png", dpi=150)
plt.clf()
print("Saved: plot_delay_by_weather.png")

# -------------------------------
# STEP 6: AVAILABILITY TIME CHECK
# -------------------------------

df["availability_time_min"].plot(
    kind="hist",
    bins=30,
    title="Ambulance Availability Time Distribution"
)

plt.xlabel("Minutes Until Available")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig("plot_availability_distribution.png", dpi=150)
plt.clf()
print("Saved: plot_availability_distribution.png")

print("\nData validation and visualization completed successfully.")
