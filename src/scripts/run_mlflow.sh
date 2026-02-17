#!/bin/bash
# -------------------------------
# start_mlflow_ui.sh
# Starts MLflow UI for local experiments
# -------------------------------

# Optional: set your tracking URI (if not default)
# export MLFLOW_TRACKING_URI="sqlite:///mnt/d/DTSC/mlflow-analyzer/data/mlflow.db"

# Start the MLflow UI on port 5000
mlflow ui --backend-store-uri sqlite:////mnt/d/DTSC/mlflow-analyzer/data/mlflow.db --port 5000
