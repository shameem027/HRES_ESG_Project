#!/bin/bash
# --- File: docker/start-mlflow.sh (NEW FILE) ---
# This script initializes the MLflow database and then starts the MLflow server.

echo "Waiting for Postgres to be fully ready before MLflow DB upgrade..."
# Loop until Postgres is ready
# Use hres_postgres as the hostname in Docker network
until pg_isready -h hres_postgres -p 5432 -U ${POSTGRES_USER} -d ${POSTGRES_DB}; do
  echo "Postgres not yet ready. Sleeping for 2 seconds..."
  sleep 2
done
echo "Postgres is ready. Running MLflow DB upgrade."

# Set MLflow tracking URI for the upgrade command
export MLFLOW_TRACKING_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@hres_postgres:5432/${POSTGRES_DB}"

# Initialize/upgrade MLflow backend database schema
# This is crucial for MLflow to correctly interact with Postgres
mlflow db upgrade ${MLFLOW_TRACKING_URI} || { echo "MLflow DB upgrade failed!"; exit 1; }
echo "MLflow DB upgrade complete. Starting MLflow server."

# Start the MLflow server
exec mlflow server \
    --backend-store-uri ${MLFLOW_TRACKING_URI} \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000