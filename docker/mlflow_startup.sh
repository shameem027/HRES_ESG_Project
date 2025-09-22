#!/bin/bash
# --- File: docker/mlflow_startup.sh ---

# This script acts as a wrapper to wait for the database to be available
# before starting the MLflow tracking server.

# Environment variables that should be passed from docker-compose.yml
# DB_HOST: The hostname of the database service (e.g., hres_postgres)
# DB_PORT: The port of the database service (e.g., 5432)
# BACKEND_STORE_URI: The full SQLAlchemy connection string for the backend store
# ARTIFACT_ROOT: The path to the artifact store

echo "Verifying database connection..."
# Loop until the database is ready to accept connections
# nc (netcat) is a simple utility to test TCP connections
until nc -z -v -w30 $DB_HOST $DB_PORT
do
  echo "Waiting for database connection at ${DB_HOST}:${DB_PORT}..."
  # wait for 5 seconds before check again
  sleep 5
done
echo "Database is up and running!"

echo "Starting MLflow Server..."
# Execute the MLflow server command
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT}