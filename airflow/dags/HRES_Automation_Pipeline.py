# --- File: airflow/dags/HRES_Automation_Pipeline.py (Definitive Automated Version) ---
from datetime import datetime, timedelta
from airflow.decorators import dag, task
import logging
import sys
import os

# Ensure Airflow can find custom modules
sys.path.append('/opt/airflow/src')
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://hres_mlflow:5000")

@dag(
    dag_id='HRES_PhD_Automation_Pipeline',
    default_args={
        'owner': 'Md_Shameem_Hossain',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Fully automated pipeline for HRES data generation, model training, and validation.',
    # --- AUTOMATION LOGIC ---
    # By setting the start_date to the past and catchup=False, the scheduler
    # will automatically trigger one run for the most recent schedule
    # as soon as it starts up. This makes the system self-initializing.
    start_date=datetime(2023, 1, 1),
    schedule='@daily',
    catchup=False,
    tags=['phd-prototype', 'hres', 'automation', 'ml'],
)
def hres_automation_pipeline():
    """
    ### HRES PhD Automation Pipeline
    This DAG orchestrates the continuous integration and validation of the HRES Decision Support Platform.
    It is configured to run automatically on system startup to initialize the dataset and ML models.
    """

    @task
    def generate_phd_level_dataset():
        """Task 1: Executes the dataset generation script."""
        logging.info("TASK 1: Kicking off dataset generation (MOO Simulation)...")
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
        """Task 2: Trains ML models and registers them in MLflow."""
        logging.info("TASK 2: Training HRES ML prediction models...")
        from HRES_ML_Model import HRESMLPredictor
        import mlflow
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            predictor = HRESMLPredictor(model_name_suffix="_V1")
            predictor.train_and_log_models(dataset_path)
            logging.info("✅ ML models training and logging successful.")
        except Exception as e:
            logging.error(f"❌ ML model training failed: {e}", exc_info=True)
            raise

    # --- Pipeline Workflow ---
    dataset_path_result = generate_phd_level_dataset()
    train_ml_models(dataset_path_result)

# Instantiate the DAG
hres_automation_pipeline()