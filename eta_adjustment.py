# ----------------------------------------
# ETA ADJUSTMENT MODULE
# Rule-based ETA adjustment using ML predictions
# ----------------------------------------

def adjust_eta(base_eta, predicted_delay, confidence, congestion, weather_severity):
    """
    Adjust base ETA using predicted delay cause, confidence, and conditions.

    Parameters:
        base_eta (float)          : Baseline ETA in minutes (distance / ideal speed)
        predicted_delay (str)     : Predicted delay cause - "Traffic", "Weather", "Incident", "No Delay"
        confidence (float)        : Model confidence score (0.0 to 1.0)
        congestion (str)          : Current congestion level - "Low", "Medium", "High"
        weather_severity (int)    : Weather severity - 0 (Clear) to 3 (Storm)

    Returns:
        float: Adjusted ETA in minutes (rounded to 2 decimal places)
    """

    adjusted_eta = base_eta

    # ----------------------------------------
    # STEP 1: APPLY DELAY CAUSE ADJUSTMENT
    # Only apply if model is confident enough
    # ----------------------------------------

    if confidence >= 0.6:

        if predicted_delay == "Traffic":
            # Traffic adds 20–50% depending on confidence
            factor = 0.20 + (0.30 * confidence)
            adjusted_eta += base_eta * factor

        elif predicted_delay == "Weather":
            # Weather adds 15–40%
            factor = 0.15 + (0.25 * confidence)
            adjusted_eta += base_eta * factor

        elif predicted_delay == "Incident":
            # Incidents cause the highest delay — 30–60%
            factor = 0.30 + (0.30 * confidence)
            adjusted_eta += base_eta * factor

        # "No Delay" — no adjustment needed

    # ----------------------------------------
    # STEP 2: APPLY CONGESTION RISK FACTOR
    # Additional buffer for high congestion
    # ----------------------------------------

    congestion_buffer = {
        "Low":    0.00,
        "Medium": 0.05,
        "High":   0.12
    }

    buffer = congestion_buffer.get(congestion, 0.0)
    adjusted_eta += base_eta * buffer

    # ----------------------------------------
    # STEP 3: APPLY WEATHER SEVERITY BUFFER
    # Small extra buffer for severe weather
    # ----------------------------------------

    if weather_severity == 1:
        adjusted_eta += base_eta * 0.03
    elif weather_severity == 2:
        adjusted_eta += base_eta * 0.07
    elif weather_severity == 3:
        adjusted_eta += base_eta * 0.12

    # ----------------------------------------
    # STEP 4: ENSURE ADJUSTED >= BASE ETA
    # Never return an ETA less than the base
    # ----------------------------------------

    adjusted_eta = max(adjusted_eta, base_eta)

    return round(adjusted_eta, 2)
