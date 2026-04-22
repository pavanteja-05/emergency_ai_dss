# ============================================================
# models.py — Pydantic Request & Response Schemas
# ============================================================

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional


# ============================================================
# SHARED VALIDATORS
# ============================================================

VALID_CONGESTION = {"Low", "Medium", "High"}
VALID_TIME_OF_DAY = {"Morning Peak", "Afternoon", "Evening Peak", "Night"}
VALID_DELAY_CAUSES = {"Traffic", "Weather", "Incident", "No Delay"}


# ============================================================
# DELAY CAUSE — Request & Response
# ============================================================

class DelayCauseRequest(BaseModel):
    avg_speed_kmph:   float = Field(..., ge=0,   le=120, description="Average speed in km/h")
    weather_severity: int   = Field(..., ge=0,   le=3,   description="0=Clear, 1=Rain, 2=Fog, 3=Storm")
    congestion_level: str   = Field(..., description="Low | Medium | High")
    time_of_day:      str   = Field(..., description="Morning Peak | Afternoon | Evening Peak | Night")
    visibility_m:     float = Field(800.0, ge=0, description="Visibility in metres (default 800)")
    speed_drop_rate:  float = Field(0.15,  ge=0, description="Speed drop rate in km/h per minute (default 0.15)")

    @field_validator("congestion_level")
    @classmethod
    def validate_congestion(cls, v):
        if v not in VALID_CONGESTION:
            raise ValueError(f"congestion_level must be one of {VALID_CONGESTION}")
        return v

    @field_validator("time_of_day")
    @classmethod
    def validate_time_of_day(cls, v):
        if v not in VALID_TIME_OF_DAY:
            raise ValueError(f"time_of_day must be one of {VALID_TIME_OF_DAY}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "avg_speed_kmph":   22.5,
                "weather_severity": 1,
                "congestion_level": "High",
                "time_of_day":      "Evening Peak",
                "visibility_m":     580.0,
                "speed_drop_rate":  0.15
            }
        }
    }


class DelayCauseResponse(BaseModel):
    delay_cause:         str               = Field(..., description="Predicted delay cause")
    confidence:          float             = Field(..., description="Model confidence (0–1)")
    class_probabilities: Dict[str, float]  = Field(..., description="Probability per class")


# ============================================================
# AVAILABILITY — Request & Response
# ============================================================

class AvailabilityRequest(BaseModel):
    expected_duration_min: float = Field(..., ge=0, description="Planned trip duration in minutes")
    distance_km:           float = Field(..., ge=0, description="Trip distance in km")
    active_trip_time_min:  float = Field(..., ge=0, description="Minutes ambulance has been on trip")
    avg_speed_kmph:        float = Field(..., ge=0, le=120, description="Current average speed")
    weather_severity:      int   = Field(..., ge=0, le=3,   description="0=Clear to 3=Storm")
    congestion_level:      str   = Field(..., description="Low | Medium | High")
    time_of_day:           str   = Field(..., description="Morning Peak | Afternoon | Evening Peak | Night")
    delay_cause:           str   = Field(..., description="Traffic | Weather | Incident | No Delay")

    @field_validator("congestion_level")
    @classmethod
    def validate_congestion(cls, v):
        if v not in VALID_CONGESTION:
            raise ValueError(f"congestion_level must be one of {VALID_CONGESTION}")
        return v

    @field_validator("time_of_day")
    @classmethod
    def validate_time_of_day(cls, v):
        if v not in VALID_TIME_OF_DAY:
            raise ValueError(f"time_of_day must be one of {VALID_TIME_OF_DAY}")
        return v

    @field_validator("delay_cause")
    @classmethod
    def validate_delay_cause(cls, v):
        if v not in VALID_DELAY_CAUSES:
            raise ValueError(f"delay_cause must be one of {VALID_DELAY_CAUSES}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "expected_duration_min": 28.5,
                "distance_km":           15.0,
                "active_trip_time_min":  18.0,
                "avg_speed_kmph":        32.0,
                "weather_severity":      1,
                "congestion_level":      "Medium",
                "time_of_day":           "Evening Peak",
                "delay_cause":           "Traffic"
            }
        }
    }


class AvailabilityResponse(BaseModel):
    availability_time_min: float = Field(..., description="Minutes until ambulance is available")
    available_at_utc:      str   = Field(..., description="Estimated UTC availability timestamp")


# ============================================================
# ETA ADJUSTMENT — Request & Response
# ============================================================

class ETARequest(BaseModel):
    base_eta_min:          float = Field(..., ge=0, description="Baseline ETA (distance / ideal speed)")
    predicted_delay_cause: str   = Field(..., description="Traffic | Weather | Incident | No Delay")
    confidence:            float = Field(..., ge=0, le=1, description="Classifier confidence score")
    congestion_level:      str   = Field(..., description="Low | Medium | High")
    weather_severity:      int   = Field(..., ge=0, le=3, description="0=Clear to 3=Storm")

    @field_validator("congestion_level")
    @classmethod
    def validate_congestion(cls, v):
        if v not in VALID_CONGESTION:
            raise ValueError(f"congestion_level must be one of {VALID_CONGESTION}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "base_eta_min":          20.0,
                "predicted_delay_cause": "Traffic",
                "confidence":            0.85,
                "congestion_level":      "High",
                "weather_severity":      1
            }
        }
    }


class ETAResponse(BaseModel):
    base_eta_min:     float = Field(..., description="Original baseline ETA")
    adjusted_eta_min: float = Field(..., description="Risk-adjusted ETA")
    added_minutes:    float = Field(..., description="Extra minutes added by risk factors")
    risk_level:       str   = Field(..., description="Low | Medium | High")


# ============================================================
# FULL PREDICTION — Request & Response
# ============================================================

class FullPredictionRequest(BaseModel):
    trip_id:               str   = Field(..., description="Unique trip identifier")
    ambulance_id:          str   = Field(..., description="Ambulance unit identifier")
    distance_km:           float = Field(..., ge=0, description="Trip distance in km")
    expected_duration_min: float = Field(..., ge=0, description="Planned trip duration in minutes")
    active_trip_time_min:  float = Field(..., ge=0, description="Minutes ambulance has been on trip")
    avg_speed_kmph:        float = Field(..., ge=0, le=120, description="Current average speed")
    weather_severity:      int   = Field(..., ge=0, le=3,   description="0=Clear to 3=Storm")
    congestion_level:      str   = Field(..., description="Low | Medium | High")
    time_of_day:           str   = Field(..., description="Morning Peak | Afternoon | Evening Peak | Night")
    visibility_m:          float = Field(800.0, ge=0, description="Visibility in metres (default 800)")
    speed_drop_rate:       float = Field(0.15,  ge=0, description="Speed drop rate in km/h per minute (default 0.15)")

    @field_validator("congestion_level")
    @classmethod
    def validate_congestion(cls, v):
        if v not in VALID_CONGESTION:
            raise ValueError(f"congestion_level must be one of {VALID_CONGESTION}")
        return v

    @field_validator("time_of_day")
    @classmethod
    def validate_time_of_day(cls, v):
        if v not in VALID_TIME_OF_DAY:
            raise ValueError(f"time_of_day must be one of {VALID_TIME_OF_DAY}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "trip_id":               "T1042",
                "ambulance_id":          "A17",
                "distance_km":           15.0,
                "expected_duration_min": 22.5,
                "active_trip_time_min":  10.0,
                "avg_speed_kmph":        28.0,
                "weather_severity":      1,
                "congestion_level":      "High",
                "time_of_day":           "Morning Peak",
                "visibility_m":          580.0,
                "speed_drop_rate":       0.15
            }
        }
    }


class FullPredictionResponse(BaseModel):
    trip_id:               str            = Field(..., description="Trip identifier")
    ambulance_id:          str            = Field(..., description="Ambulance identifier")
    timestamp:             str            = Field(..., description="UTC timestamp of prediction")
    delay_cause:           str            = Field(..., description="Predicted delay cause")
    confidence:            float          = Field(..., description="Classifier confidence")
    class_probabilities:   Dict[str, float] = Field(..., description="Probability per delay class")
    availability_time_min: float          = Field(..., description="Minutes until ambulance available")
    base_eta_min:          float          = Field(..., description="Baseline ETA in minutes")
    adjusted_eta_min:      float          = Field(..., description="Risk-adjusted ETA in minutes")
    added_minutes:         float          = Field(..., description="Extra delay added by risk factors")
    risk_level:            str            = Field(..., description="Overall ETA risk: Low|Medium|High")


# ============================================================
# LIVE SIMULATION — Response
# ============================================================

class LiveTripResponse(BaseModel):
    trip_id:               str   = Field(..., description="Simulated trip ID")
    ambulance_id:          str   = Field(..., description="Simulated ambulance ID")
    distance_km:           float = Field(..., description="Trip distance in km")
    expected_duration_min: float = Field(..., description="Expected duration in minutes")
    active_trip_time_min:  float = Field(..., description="Time ambulance has been on trip")
    avg_speed_kmph:        float = Field(..., description="Average speed")
    congestion_level:      str   = Field(..., description="Congestion level")
    weather_condition:     str   = Field(..., description="Weather description")
    weather_severity:      int   = Field(..., description="Weather severity 0–3")
    time_of_day:           str   = Field(..., description="Time period")
    timestamp:             str   = Field(..., description="UTC timestamp")
