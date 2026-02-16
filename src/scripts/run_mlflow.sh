#!/bin/bash
# -------------------------------
# start_mlflow_ui.sh
# Starts MLflow UI for local experiments
# -------------------------------

# Optional: set your tracking URI (if not default)
# export MLFLOW_TRACKING_URI="../../data/mlruns"

# Start the MLflow UI on port 5000
mlflow ui --backend-store-uri /mnt/d/DTSC/mlflow-analyzer/data/mlruns --port 5000
