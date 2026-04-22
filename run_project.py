#!/usr/bin/env python3
"""
AI Emergency Support System — One-Shot Setup & Launch Script
============================================================
Run this from the project root to:
  1. Install all dependencies
  2. Train both ML models (if .pkl files missing)
  3. Launch the Streamlit dashboard

Usage:
    python run_project.py
    python run_project.py --dashboard-only   (skip training, just launch)
    python run_project.py --train-only       (train models, don't launch)
"""

import os, sys, subprocess, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")

REQUIRED_PKLS = [
    "delay_cause_model.pkl", "le_congestion.pkl",
    "le_time_of_day.pkl",    "le_delay_cause.pkl",
    "availability_model.pkl","avail_le_congestion.pkl",
    "avail_le_time_of_day.pkl","avail_le_delay_cause.pkl"
]

def banner(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")

def run(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)

def install_deps():
    banner("STEP 1 — Installing dependencies")
    run(f"pip install -r {os.path.join(BACKEND, 'requirements.txt')}")
    print("[OK] Dependencies installed")

def train_models():
    banner("STEP 2 — Training ML models")
    missing = [p for p in REQUIRED_PKLS if not os.path.exists(os.path.join(BACKEND, p))]
    if not missing:
        print("[OK] All model .pkl files already exist — skipping training")
        return

    print(f"Missing: {missing}")
    print("Training delay cause classifier...")
    run("python delay_cause_model.py", cwd=ROOT)
    print("Training availability predictor...")
    run("python availability_model.py", cwd=ROOT)

    # Copy .pkl files into backend/
    import glob, shutil
    for pkl in glob.glob(os.path.join(ROOT, "*.pkl")):
        shutil.copy(pkl, BACKEND)
        print(f"  Copied {os.path.basename(pkl)} → backend/")

    print("[OK] Models trained and saved")

def launch_dashboard():
    banner("STEP 3 — Launching Streamlit dashboard")
    print("Dashboard URL: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    os.system(f"streamlit run {os.path.join(BACKEND, 'dashboard.py')} --server.port 8501")

def launch_api():
    """Optional: run the FastAPI backend in a separate terminal."""
    banner("FastAPI Backend")
    print("API URL:       http://localhost:8000")
    print("API Docs:      http://localhost:8000/docs")
    os.system(f"cd {BACKEND} && uvicorn main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Emergency DSS — Setup & Launch")
    parser.add_argument("--dashboard-only", action="store_true", help="Skip training, just launch dashboard")
    parser.add_argument("--train-only",     action="store_true", help="Train models only, don't launch")
    parser.add_argument("--api",            action="store_true", help="Launch FastAPI backend instead of dashboard")
    args = parser.parse_args()

    if args.dashboard_only:
        launch_dashboard()
    elif args.train_only:
        install_deps()
        train_models()
    elif args.api:
        launch_api()
    else:
        install_deps()
        train_models()
        launch_dashboard()
