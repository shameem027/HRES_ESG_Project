# --- File: airflow/dags/HRES_Automation_Pipeline.py ---
from datetime import datetime, timedelta
from airflow.decorators import dag, task
import logging
import sys
import pandas as pd
import mlflow
import os

# --- Configuration ---
# This path is essential for Airflow to find custom modules in the 'src' directory
sys.path.append('/opt/airflow/src')
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://hres_mlflow:5000")


@dag(
    dag_id='HRES_PhD_Automation_Pipeline',
    default_args={
        'owner': 'Md_Shameem_Hossain',
        'depends_on_past': False,
        'start_date': datetime(2025, 9, 21),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=2),
    },
    description='Automates the generation, validation, and ML model training for the HRES recommender system.',
    schedule='@weekly',
    catchup=False,
    tags=['phd-prototype', 'hres', 'esg', 'mcda', 'automation', 'ml'],
)
def hres_automation_pipeline():
    """
    ### HRES PhD Automation Pipeline
    This DAG orchestrates the continuous integration and validation of the HRES Decision Support Platform.
    It simulates a full MLOps lifecycle for the research prototype:
    1.  **Data Generation (MOO):** Creates a fresh, realistic dataset of HRES solutions.
    2.  **ML Model Training:** Trains predictive ML models on the new dataset and registers them in MLflow.
    3.  **Logic Validation (ELECTRE TRI & MCDA):** Tests the decision engine against benchmark scenarios.
    4.  **Reporting:** Logs all results to MLflow and provides a completion notification.
    """

    @task
    def generate_phd_level_dataset():
        """Task 1: Executes the dataset generation script (MOO phase)."""
        logging.info("TASK 1: Kicking off dataset generation (MOO Simulation)...")
        # Defer import until runtime to ensure sys.path is set
        from HRES_Dataset_Generator import main as generate_dataset
        try:
            generate_dataset()
            dataset_path = "/opt/airflow/src/HRES_Dataset.csv"
            logging.info(f"✅ Dataset generation successful. Path: {dataset_path}")
            return dataset_path
        except Exception as e:
            logging.error(f"❌ Dataset generation failed: {e}", exc_info=True)
            raise

    @task
    def train_ml_models(dataset_path: str):
        """Task 2: Trains ML models for HRES outcome prediction and logs them to MLflow."""
        logging.info("TASK 2: Training HRES ML prediction models...")
        from HRES_ML_Model import HRESMLPredictor
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            predictor = HRESMLPredictor(model_name_suffix="_V1")
            predictor.train_and_log_models(dataset_path)
            logging.info("✅ ML models training and logging successful.")
        except Exception as e:
            logging.error(f"❌ ML model training failed: {e}", exc_info=True)
            raise

    @task
    def validate_mcda_logic_with_benchmarks(dataset_path: str):
        """Task 3: Executes the MCDA model against benchmarks and logs results to MLflow."""
        logging.info("TASK 3: Running benchmark validation scenarios (MCDA & ELECTRE TRI Simulation)...")
        from MCDA_model import HRES_Decision_Engine
        try:
            df = pd.read_csv(dataset_path)
            decision_engine = HRES_Decision_Engine(df)
            logging.info("✅ Decision Engine initialized successfully.")

            scenarios = {
                "Cost_Focus_Office": {
                    "input": {"scenario_name": "Small_Office", "annual_demand_kwh": 250000,
                              "user_grid_dependency_pct": 30,
                              "esg_weights": {"cost": 0.9, "environment": 0.1, "social": 0.0, "governance": 0.0}},
                    "expected_payback_max": 20.0
                },
                "Resilience_Focus_Hospital": {
                    "input": {"scenario_name": "Hospital", "annual_demand_kwh": 1500000, "user_grid_dependency_pct": 10,
                              "esg_weights": {"cost": 0.1, "environment": 0.3, "social": 0.6, "governance": 0.0}},
                    "expected_resilience_min": 16.0
                }
            }
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("HRES_Automated_Validation")

            with mlflow.start_run(run_name=f"Weekly_Validation_{datetime.now().strftime('%Y-%m-%d')}") as run:
                mlflow.log_artifact(dataset_path, "datasets")
                validation_results = {}
                overall_success = True

                for name, params in scenarios.items():
                    logging.info(f"--- Validating scenario: {name} ---")
                    solution, status, _, _, _ = decision_engine.run_full_pipeline(**params["input"])

                    if solution is None:
                        logging.error(f"❌ Validation for '{name}' failed to find a solution. Status: {status}")
                        validation_results[f"{name}_passed"] = 0
                        overall_success = False
                        continue

                    payback_check = solution.get('payback_period_years', float('inf')) <= params['expected_payback_max']
                    resilience_check = solution.get('soc_energy_resilience_hrs', 0.0) >= params[
                        'expected_resilience_min']

                    passed = payback_check and resilience_check
                    validation_results[f"{name}_passed"] = 1 if passed else 0
                    logging.info(f"✅ Scenario '{name}' validation result: {'PASSED' if passed else 'FAILED'}. "
                                 f"Payback: {solution.get('payback_period_years'):.1f} yrs, "
                                 f"Resilience: {solution.get('soc_energy_resilience_hrs'):.1f} hrs.")
                    if not passed:
                        overall_success = False

                mlflow.log_metrics(validation_results)
                mlflow.log_metric("overall_validation_passed", 1 if overall_success else 0)
                if not overall_success:
                    raise ValueError("One or more validation benchmarks failed.")
            return run.info.run_id
        except Exception as e:
            logging.error(f"❌ MCDA validation failed: {e}", exc_info=True)
            raise

    @task
    def send_completion_notification(mlflow_run_id: str):
        """Task 4: Sends a completion notification with a link to the results."""
        logging.info("TASK 4: Sending pipeline completion notification.")
        # In a real-world scenario, this would use an EmailOperator or SlackOperator.
        # For this project, logging is sufficient.
        print("=" * 80)
        print("🎉 HRES MLOps Automation Pipeline Completed Successfully!")
        print(f"Timestamp: {datetime.now()}")
        print(f"📊 View full validation results in MLflow:")
        print(f"   {MLFLOW_TRACKING_URI}/#/experiments/1/runs/{mlflow_run_id}")
        print("=" * 80)

    # --- Define the pipeline's workflow ---
    dataset_path_result = generate_phd_level_dataset()
    # ML training and MCDA validation can run in parallel after dataset generation
    ml_training_task = train_ml_models(dataset_path_result)
    mcda_validation_task = validate_mcda_logic_with_benchmarks(dataset_path_result)

    # The notification task depends on the successful completion of the MCDA validation
    send_completion_notification(mcda_validation_task)


# Instantiate the DAG
hres_automation_pipeline()