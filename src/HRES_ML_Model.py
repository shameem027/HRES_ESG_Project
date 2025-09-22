# --- File: src/HRES_ML_Model.py ---
# --- Author: Md Shameem Hossain ---
# --- Purpose: Trains and logs Machine Learning models to predict HRES outcomes for faster inference. ---

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow setup will be handled by the environment (Airflow DAG or API container)
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://hres_mlflow:5000")
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("HRES_ML_Prediction_Models")


class HRESMLPredictor:
    def __init__(self, model_name_suffix=""):
        self.models = {}
        self.model_name_suffix = model_name_suffix
        self.target_cols = ['total_cost', 'self_sufficiency_pct', 'annual_savings_eur']
        self.feature_cols = ['num_solar_panels', 'num_wind_turbines', 'battery_kwh']
        self.scenario_encoder = {}

    def _prepare_data(self, df):
        scenario_names = df['scenario_name'].unique()
        self.scenario_encoder = {name: i for i, name in enumerate(scenario_names)}

        df_encoded = df.copy()
        df_encoded['scenario_encoded'] = df_encoded['scenario_name'].map(self.scenario_encoder)

        features = self.feature_cols + ['scenario_encoded']
        X = df_encoded[features]
        y = df_encoded[self.target_cols]
        return X, y

    def train_and_log_models(self, dataset_path: str):
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Training ML models on dataset with {len(df)} rows.")
        except FileNotFoundError:
            logger.error(f"Dataset not found at {dataset_path}. Skipping ML model training.")
            return

        X, y = self._prepare_data(df)

        for target in self.target_cols:
            with mlflow.start_run(run_name=f"RandomForest_{target}{self.model_name_suffix}") as run:
                logger.info(f"Training model for target: {target}")

                params = {"n_estimators": 100, "max_depth": 10, "random_state": 42,
                          "min_samples_split": 5, "min_samples_leaf": 3}

                model = RandomForestRegressor(**params)
                model.fit(X, y[target])

                y_pred = model.predict(X)
                mae = mean_absolute_error(y[target], y_pred)
                r2 = r2_score(y[target], y_pred)

                logger.info(f"Target: {target}, MAE: {mae:.2f}, R2: {r2:.2f}")

                mlflow.log_params(params)
                mlflow.log_metrics({"mae": mae, "r2": r2})

                # Log the scenario encoder with the model for consistent inference
                mlflow.log_dict(self.scenario_encoder, "scenario_encoder.json")

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"hres_ml_model_{target}",
                    registered_model_name=f"HRES_ML_Predictor_{target}{self.model_name_suffix}"
                )
                self.models[target] = model
        logger.info("✅ ML models training and logging complete.")

    @staticmethod
    def load_latest_model(target: str, model_name_suffix=""):
        model_name = f"HRES_ML_Predictor_{target}{model_name_suffix}"
        client = mlflow.tracking.MlflowClient()
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if not latest_versions:
                raise Exception("No versions found for this model name.")

            latest_version_obj = latest_versions[0]
            model_uri = f"models:/{model_name}/{latest_version_obj.version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(
                f"✅ Successfully loaded ML model '{model_name}' v{latest_version_obj.version} from '{model_uri}'")

            # Download the associated encoder artifact
            run_id = latest_version_obj.run_id
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scenario_encoder.json")
            with open(local_path, 'r') as f:
                scenario_encoder = json.load(f)
            logger.info(f"✅ Loaded associated scenario encoder: {scenario_encoder}")

            return model, scenario_encoder
        except Exception as e:
            logger.error(f"❌ Failed to load ML model '{model_name}': {e}", exc_info=True)
            return None, None


def main():
    # This allows the script to be run manually for testing/debugging
    # It assumes it's being run from the project root or the `src` directory
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, 'HRES_Dataset.csv')
    predictor = HRESMLPredictor(model_name_suffix="_V1")
    predictor.train_and_log_models(dataset_path)


if __name__ == "__main__":
    main()
