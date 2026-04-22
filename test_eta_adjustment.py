from eta_adjustment import adjust_eta

# Base ETA without risk
base_eta = 20

# Simulated ML predictions & conditions
predicted_delay = "Traffic"
confidence = 0.85
congestion = "High"
weather_sev = 1

# Adjust ETA using rule-based logic
final_eta = adjust_eta(
    base_eta,
    predicted_delay,
    confidence,
    congestion,
    weather_sev
)

# Display results
print("Base ETA:", base_eta)
print("Adjusted ETA:", final_eta)
