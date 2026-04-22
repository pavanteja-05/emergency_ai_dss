# 🚑 AI-Driven Emergency Response Assistant

> A simulation-driven AI Decision Support System for emergency ambulance dispatch.  
> Predicts delay causes, ambulance availability, and risk-adjusted ETAs in real time.

---

## Overview

This project uses synthetically generated traffic, weather, and trip data to train machine learning models that assist emergency dispatch decisions. All predictions are visualised in a live Streamlit dashboard.

---

## Key Features

- **Delay Cause Classifier** — predicts Traffic / Weather / Incident / No Delay (91.7% accuracy)
- **Availability Predictor** — estimates minutes until ambulance is free (R² = 0.91)
- **ETA Adjustment** — rule-based risk adjustment using delay cause and conditions
- **Real-Time Dashboard** — live simulation with trend charts and confidence gauges
- **FastAPI Backend** — 6 REST endpoints for external integration

---

## Technology Stack

| Component        | Technology                          |
|-----------------|-------------------------------------|
| Language         | Python 3.10+                        |
| Machine Learning | scikit-learn (Random Forest)        |
| Dashboard        | Streamlit, Plotly                   |
| API Backend      | FastAPI, Uvicorn, Pydantic          |
| Data Processing  | pandas, numpy, joblib               |

---

## Project Structure

```
emergency_ai_dss/
│
├── trip_time_simulation.py      
├── traffic_simulation.py      
├── weather_simulation.py     
├── delay_simulation.py      
├── delay_cause_model.py         
├── availability_model.py         
├── eta_adjustment.py            
├── data_validation.py           
├── final_simulated_emergency_data.csv
├── run_project.py             
│
└── backend/
    ├── dashboard.py              ← Streamlit dashboard
    ├── main.py                   ← FastAPI app
    ├── models.py                 ← Pydantic schemas
    ├── eta_adjustment.py
    ├── requirements.txt
    └── README.md
```

---

## Setup and Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2: Generate Simulated Dataset
*Skip if dataset already exists*
```bash
python trip_time_simulation.py
python traffic_simulation.py
python weather_simulation.py
python delay_simulation.py
```

### Step 3: Train Delay Cause Classifier
*Generates 5 .pkl files*
```bash
python delay_cause_model.py
```

### Step 4: Train Availability Predictor
*Generates 4 .pkl files*
```bash
python availability_model.py
```

### Step 5: Move Model Files to Backend
```bash
cp *.pkl backend/
```

### Step 6: Launch FastAPI Backend (Terminal 1)
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 7: Launch Streamlit Dashboard (Terminal 2)
```bash
cd backend
streamlit run dashboard.py --server.port 8501
```

## Access Points

- Dashboard: http://localhost:8501  
- API Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

## Notes

- Use Python 3.10 or above  
- Recommended to use virtual environment  
- Ensure all `.pkl` files are inside `backend/` before running  

### API Endpoints

| Method | Endpoint                  | Description                          |
|--------|--------------------------|--------------------------------------|
| GET    | `/health`                | System status                        |
| POST   | `/predict/delay-cause`   | Classify delay cause                 |
| POST   | `/predict/availability`  | Predict availability time            |
| POST   | `/predict/eta`           | Get risk-adjusted ETA                |
| POST   | `/predict/full`          | All three predictions in one call    |
| GET    | `/simulate/live-trip`    | Generate a random simulated trip     |

---

