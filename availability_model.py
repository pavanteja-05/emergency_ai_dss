# ----------------------------------------
# AMBULANCE AVAILABILITY PREDICTOR
# Predicts minutes until an ambulance
# becomes available after completing a trip.
#
# DESIGN NOTE — Realistic Feature Selection:
# At prediction time, a dispatcher knows:
#   - distance_km, expected_duration_min    (trip plan — known upfront)
#   - active_trip_time_min                 (how long busy — real-time)
#   - congestion, weather, time_of_day     (current conditions)
#   - delay_cause                          (from the delay classifier)
#
# Excluded to avoid data leakage:
#   - actual_duration_min (not known until trip ends)
#   - delay_minutes       (actual - expected, so also unknown upfront)
# ----------------------------------------

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------
# STEP 1: LOAD DATASET
# ----------------------------------------

df = pd.read_csv("final_simulated_emergency_data.csv")
print("Dataset loaded successfully")
print("Total rows:", len(df))

# ----------------------------------------
# STEP 2: DROP NULLS (safety guard)
# ----------------------------------------

before = len(df)
df = df.dropna(subset=["availability_time_min", "active_trip_time_min", "delay_cause"])
after = len(df)
if before != after:
    print(f"Warning: Dropped {before - after} rows with null values")

# ----------------------------------------
# STEP 3: ENCODE CATEGORICAL FEATURES
# ----------------------------------------

le_congestion  = LabelEncoder()
le_time_of_day = LabelEncoder()
le_delay_cause = LabelEncoder()

df["congestion_level_enc"] = le_congestion.fit_transform(df["congestion_level"])
df["time_of_day_enc"]      = le_time_of_day.fit_transform(df["time_of_day"])
df["delay_cause_enc"]      = le_delay_cause.fit_transform(df["delay_cause"])

# ----------------------------------------
# STEP 4: FEATURE ENGINEERING
# ----------------------------------------

# time_pressure: ratio of active trip time to expected duration
# Captures how far along the trip the ambulance is (0 = just dispatched, 1+ = overdue)
df["time_pressure"] = (
    df["active_trip_time_min"] / df["expected_duration_min"]
).clip(0, 3)

# congestion_weather_risk: combined stress factor on the trip
df["congestion_weather_risk"] = df["weather_severity"] * (df["congestion_level_enc"] + 1)

# ----------------------------------------
# STEP 5: DEFINE FEATURES & TARGET
# ----------------------------------------

feature_columns = [
    "expected_duration_min",    # planned trip length (known upfront)
    "distance_km",              # trip distance (known upfront)
    "active_trip_time_min",     # how long ambulance has been busy (real-time)
    "time_pressure",            # engineered: progress ratio through trip
    "avg_speed_kmph",           # observed average speed (real-time)
    "weather_severity",         # current weather severity
    "congestion_weather_risk",  # engineered: combined stress factor
    "congestion_level_enc",     # congestion level encoding
    "time_of_day_enc",          # time period encoding
    "delay_cause_enc"           # from delay cause classifier output
]

target_column = "availability_time_min"

X = df[feature_columns]
y = df[target_column]

print("\nFeatures used:")
for f in feature_columns:
    print(f"  - {f}")
print("Target:", target_column)

# ----------------------------------------
# STEP 6: TRAIN–TEST SPLIT
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ----------------------------------------
# STEP 7: TRAIN BOTH MODELS & COMPARE
# ----------------------------------------

# --- Linear Regression (baseline) ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)

print("\n--- Linear Regression (Baseline) ---")
print(f"  MAE  : {lr_mae:.2f} minutes")
print(f"  RMSE : {lr_rmse:.2f} minutes")
print(f"  R2   : {lr_r2:.4f}")

# --- Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)

print("\n--- Random Forest Regressor ---")
print(f"  MAE  : {rf_mae:.2f} minutes")
print(f"  RMSE : {rf_rmse:.2f} minutes")
print(f"  R2   : {rf_r2:.4f}")

# ----------------------------------------
# STEP 8: SELECT BEST MODEL
# ----------------------------------------

if rf_r2 >= lr_r2:
    best_model, best_model_name = rf_model, "Random Forest Regressor"
    best_pred, best_r2, best_mae, best_rmse = rf_pred, rf_r2, rf_mae, rf_rmse
else:
    best_model, best_model_name = lr_model, "Linear Regression"
    best_pred, best_r2, best_mae, best_rmse = lr_pred, lr_r2, lr_mae, lr_rmse

print(f"\nBest Model Selected: {best_model_name}")
print(f"  R2   = {best_r2:.4f}")
print(f"  MAE  = {best_mae:.2f} minutes")
print(f"  RMSE = {best_rmse:.2f} minutes")

# ----------------------------------------
# STEP 9: CROSS-VALIDATION (5-FOLD)
# ----------------------------------------

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"\n5-Fold CV R2 Scores : {cv_scores.round(4)}")
print(f"Mean CV R2          : {cv_scores.mean():.4f}  +/-  {cv_scores.std():.4f}")

# ----------------------------------------
# STEP 10: FEATURE IMPORTANCE (RF only)
# ----------------------------------------

if best_model_name == "Random Forest Regressor":
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)

    print("\nFeature Importances:")
    for feat, imp in importances.items():
        print(f"  {feat:<30} {imp:.4f}")

    importances.plot(
        kind="bar",
        title="Feature Importances - Availability Predictor",
        color="steelblue"
    )
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plot_availability_feature_importance.png", dpi=150)
    plt.clf()
    print("\nSaved: plot_availability_feature_importance.png")

# ----------------------------------------
# STEP 11: ACTUAL vs PREDICTED PLOT
# ----------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, alpha=0.3, color="steelblue", edgecolors="none")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red", linewidth=1.5, label="Perfect prediction"
)
plt.xlabel("Actual Availability Time (min)")
plt.ylabel("Predicted Availability Time (min)")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig("plot_availability_actual_vs_predicted.png", dpi=150)
plt.clf()
print("Saved: plot_availability_actual_vs_predicted.png")

# ----------------------------------------
# STEP 12: SAVE MODEL & ENCODERS
# ----------------------------------------

joblib.dump(best_model,     "availability_model.pkl")
joblib.dump(le_congestion,  "avail_le_congestion.pkl")
joblib.dump(le_time_of_day, "avail_le_time_of_day.pkl")
joblib.dump(le_delay_cause, "avail_le_delay_cause.pkl")

print("\nModel and encoders saved:")
print("  - availability_model.pkl")
print("  - avail_le_congestion.pkl")
print("  - avail_le_time_of_day.pkl")
print("  - avail_le_delay_cause.pkl")

# ----------------------------------------
# STEP 13: QUICK INFERENCE DEMO
# ----------------------------------------

print("\n--- Quick Inference Demo ---")

exp_dur = 28.5
active  = 18.0
speed   = 32.0
cong    = "Medium"
tod     = "Evening Peak"
dc      = "Traffic"
wsev    = 1

cong_enc = le_congestion.transform([cong])[0]
tod_enc  = le_time_of_day.transform([tod])[0]
dc_enc   = le_delay_cause.transform([dc])[0]

sample = pd.DataFrame([{
    "expected_duration_min":   exp_dur,
    "distance_km":             15.0,
    "active_trip_time_min":    active,
    "time_pressure":           min(active / exp_dur, 3),
    "avg_speed_kmph":          speed,
    "weather_severity":        wsev,
    "congestion_weather_risk": wsev * (cong_enc + 1),
    "congestion_level_enc":    cong_enc,
    "time_of_day_enc":         tod_enc,
    "delay_cause_enc":         dc_enc
}])

predicted = best_model.predict(sample)[0]
print(f"  Ambulance is {active} min into an expected {exp_dur} min trip")
print(f"  Conditions : {dc} delay | {cong} congestion | {tod} | Weather severity {wsev}")
print(f"  Predicted availability in: {predicted:.1f} minutes")
