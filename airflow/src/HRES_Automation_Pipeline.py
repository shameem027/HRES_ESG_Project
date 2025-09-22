# --- File: airflow/dags/HRES_Automation_Pipeline.py (PhD-Level Definitive Version) ---
# --- Author: Md Shameem Hossain ---
# --- Purpose: An Airflow DAG to automate the HRES MLOps pipeline, reflecting the PhD research framework ---

from datetime import datetime, timedelta
from airflow.decorators import dag, task
import logging
import sys
import pandas as pd
import mlflow

# --- Configuration ---
sys.path.append('/opt/airflow/src')
MLFLOW_TRACKING_URI = "http://mlflow:5000"


@dag(
    dag_id='HRES_PhD_Automation_Pipeline',
    default_args={
        'owner': 'Md_Shameem_Hossain', 'depends_on_past': False, 'start_date': datetime(2025, 9, 21),
        'email_on_failure': False, 'email_on_retry': False, 'retries': 1, 'retry_delay': timedelta(minutes=2),
    },
    description='Automates the generation and validation of the HRES recommender system dataset and logic.',
    schedule='@weekly',
    catchup=False,
    tags=['phd-prototype', 'hres', 'esg', 'mcda', 'automation'],
)
def hres_automation_pipeline():
    """
    ### HRES PhD Automation Pipeline
    This DAG orchestrates the continuous validation of the HRES Decision Support Platform, simulating the full MLOps lifecycle.
    """

    @task
    def generate_phd_level_dataset():
        """Task 1: Executes the PhD-level dataset generation script (MOO phase)."""
        logging.info("TASK 1: Kicking off PhD-level dataset generation (MOO Simulation)...")
        from HRES_Dataset_Generator import main as generate_dataset
        try:
            generate_dataset()
            logging.info("✅ Dataset generation successful.")
            return "/opt/airflow/src/HRES_Dataset.csv"
        except Exception as e:
            logging.error(f"❌ Dataset generation failed: {e}")
            raise

    @task
    def validate_mcda_logic_with_benchmarks(dataset_path: str):
        """Task 2: Executes the MCDA model against benchmarks and logs to MLflow."""
        logging.info("TASK 2: Running benchmark validation scenarios...")
        from MCDA_model import HRES_Decision_Engine
        try:
            df = pd.read_csv(dataset_path)
            decision_engine = HRES_Decision_Engine(df)
            logging.info("✅ Decision Engine initialized successfully.")

            scenarios = {
                "Cost_Focus_Office": {
                    "input": {"scenario_name": "Small_Office", "annual_demand_kwh": 250000, "user_grid_dependency_pct": 30,
                              "esg_weights": {"cost": 0.9, "environment": 0.1, "social": 0.0, "governance": 0.0}}, # FIX: "environmental" -> "environment"
                    "expected_cost_max": 200000
                },
                "Resilience_Focus_Hospital": {
                    "input": {"scenario_name": "Hospital", "annual_demand_kwh": 1500000, "user_grid_dependency_pct": 10,
                              "esg_weights": {"cost": 0.1, "environment": 0.3, "social": 0.6, "governance": 0.0}}, # FIX: "environmental" -> "environment"
                    "expected_resilience_min": 8.0
                }
            }

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("HRES_Automated_Validation")
            with mlflow.start_run(run_name=f"Weekly_Validation_{datetime.now().strftime('%Y-%m-%d')}") as run:
                mlflow.log_artifact(dataset_path, "datasets")
                validation_results = {}
                for name, params in scenarios.items():
                    logging.info(f"--- Validating scenario: {name} ---")
                    solution, _, _, _, _ = decision_engine.run_full_pipeline(**params["input"])
                    if solution is None:
                        logging.error(f"❌ Validation for '{name}' failed to find a solution.")
                        validation_results[f"{name}_passed"] = 0; continue
                    cost_check = solution['total_cost'] < params.get('expected_cost_max', float('inf'))
                    resilience_check = solution['soc_energy_resilience_hrs'] >= params.get('expected_resilience_min', 0)
                    passed = cost_check and resilience_check
                    validation_results[f"{name}_passed"] = 1 if passed else 0
                    logging.info(f"✅ Scenario '{name}' validation result: {'PASSED' if passed else 'FAILED'}")

                overall_success = all(v == 1 for v in validation_results.values())
                mlflow.log_metrics(validation_results)
                mlflow.log_metric("overall_validation_passed", 1 if overall_success else 0)
                if not overall_success: raise ValueError("One or more validation benchmarks failed.")
            return run.info.run_id
        except Exception as e:
            logging.error(f"❌ MCDA validation failed: {e}")
            raise

    @task
    def send_completion_notification(mlflow_run_id: str):
        """Task 3: Sends a completion notification with a link to the results."""
        logging.info("TASK 3: Sending pipeline completion notification.")
        print("="*60)
        print("🎉 HRES MLOps Automation Pipeline Completed Successfully!")
        print(f"Timestamp: {datetime.now()}")
        print(f"📊 View full results in MLflow: {MLFLOW_TRACKING_URI}/#/experiments/2/runs/{mlflow_run_id}")
        print("="*60)

    # --- Define the pipeline's workflow ---
    dataset_path_result = generate_phd_level_dataset()
    mlflow_run_id_result = validate_mcda_logic_with_benchmarks(dataset_path_result)
    send_completion_notification(mlflow_run_id_result)

hres_automation_pipeline()