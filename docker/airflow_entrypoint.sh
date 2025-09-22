#!/bin/bash
# --- File: docker/airflow_entrypoint.sh (Cleaned Version) ---

# Define default values for Airflow connections
AIRFLOW_CONN_POSTGRES_DEFAULT='postgres://airflow:airflow@hres_postgres:5432/airflow'

# Export the connection if it's not already set
export AIRFLOW_CONN_POSTGRES_DEFAULT

# Wait for the database to be available
echo "Waiting for PostgreSQL..."
while ! nc -z hres_postgres 5432; do
  sleep 0.1
done
echo "PostgreSQL started"

# Run the database migrations
echo "Running Airflow DB migrations..."
airflow db upgrade

# Create the admin user if not exists
echo "Checking/Creating Airflow admin user..."
airflow users create \
    --username "${AIRFLOW_WWW_USER_USERNAME:-airflow}" \
    --password "${AIRFLOW_WWW_USER_PASSWORD:-airflow}" \
    --firstname Shameem \
    --lastname Hossain \
    --role Admin \
    --email shameem.hossain@email.com || true

# The 'exec' command replaces the script process with the Airflow process
exec "$@"