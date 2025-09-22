# --- File: api/recommender_api.py (Definitive Final Version) ---
import os
import pandas as pd
from flask import Flask, request, jsonify
import logging
import json
from openai import AzureOpenAI
import mlflow
import mlflow.sklearn
from pydantic import BaseModel, ValidationError, Field
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ... (Pydantic models and Prompts are correct and remain the same) ...
class EsgWeights(BaseModel):
    environment: float = Field(..., ge=0, le=1); social: float = Field(..., ge=0, le=1); governance: float = Field(..., ge=0, le=1); cost: float = Field(..., ge=0, le=1)
class RecommendRequest(BaseModel):
    scenario_name: str; annual_demand_kwh: int = Field(..., gt=0); user_grid_dependency_pct: int = Field(..., ge=0, le=100); esg_weights: EsgWeights
class MlPredictRequest(BaseModel):
    scenario_name: str; num_solar_panels: int = Field(..., ge=0); num_wind_turbines: int = Field(..., ge=0); battery_kwh: int = Field(..., ge=0)
INTENT_PARSING_PROMPT = """...""" # (Same as before)
RESPONSE_GENERATION_PROMPT = """...""" # (Same as before)

# --- Global Variables ---
decision_engine = None; ml_models = {}; scenario_encoders = {}; azure_client = None

# --- Robust Resource Loading ---
def load_resources():
    global decision_engine, ml_models, scenario_encoders, azure_client
    logger.info("Attempting to load resources...")
    try:
        from src.MCDA_model import HRES_Decision_Engine
        DATA_PATH = "/app/src/HRES_Dataset.csv"
        hres_df = pd.read_csv(DATA_PATH)
        decision_engine = HRES_Decision_Engine(hres_df)
        logger.info(f"✅ HRES Decision Engine initialized.")
    except Exception as e: logger.warning(f"⚠️ Decision Engine not yet loaded. Waiting for data. Error: {e}")
    try:
        from src.HRES_ML_Model import HRESMLPredictor
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        for target in ['total_cost', 'self_sufficiency_pct', 'annual_savings_eur']:
            model, encoder = HRESMLPredictor.load_latest_model(target, model_name_suffix="_V1")
            if model and encoder: ml_models[target] = model; scenario_encoders[target] = encoder
        if ml_models: logger.info(f"✅ Loaded {len(ml_models)} ML models.")
        else: logger.warning("⚠️ ML models not yet loaded. Waiting for training.")
    except Exception as e: logger.warning(f"⚠️ Could not load ML models. Error: {e}")
    try:
        if not azure_client and os.getenv("AZURE_OPENAI_API_KEY"):
            azure_client = AzureOpenAI(api_version=os.getenv("OPENAI_API_VERSION"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_key=os.getenv("AZURE_OPENAI_API_KEY"))
            logger.info("✅ Azure OpenAI client initialized.")
    except Exception as e: logger.warning(f"⚠️ Could not initialize Azure OpenAI client. Error: {e}.")

@app.before_request
def before_request_func():
    # This hook ensures resources are re-checked before each request if they failed to load initially
    if not decision_engine or not ml_models:
        load_resources()

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "decision_engine_loaded": decision_engine is not None, "llm_client_loaded": azure_client is not None, "ml_models_loaded": len(ml_models) > 0})

# ... (All other endpoints: /recommend, /predict_ml, /chat are correct and remain the same) ...
@app.route('/recommend', methods=['POST'])
def recommend():
    if not decision_engine: return jsonify({"error": "Decision Engine not operational. Data may still be generating."}), 503
    try: req_data = RecommendRequest.model_validate(request.get_json()); best_solution, status, feasible_df, sorted_df, pareto_df = decision_engine.run_full_pipeline(req_data.scenario_name, req_data.annual_demand_kwh, req_data.user_grid_dependency_pct, req_data.esg_weights.model_dump());
    except Exception as e: return jsonify({"error": "Internal server error"}), 500
    if best_solution is not None: return jsonify({"status": status, "recommendation": json.loads(pd.Series(best_solution).to_json()), "intermediate_results": {"pareto_front": pareto_df.to_dict(orient='records') if pareto_df is not None else []}}), 200
    else: return jsonify({"status": status, "recommendation": None}), 404
@app.route('/predict_ml', methods=['POST'])
def predict_ml():
    if not ml_models: return jsonify({"error": "ML models not loaded. Please wait for the automated training pipeline to complete."}), 503
    try: req_data = MlPredictRequest.model_validate(request.get_json()); encoder = scenario_encoders.get('total_cost')
    except Exception as e: return jsonify({"error": "Internal server error"}), 500
    if not encoder or req_data.scenario_name not in encoder: return jsonify({"error": "Scenario not recognized"}), 400
    scenario_encoded = encoder[req_data.scenario_name]; input_df = pd.DataFrame([[req_data.num_solar_panels, req_data.num_wind_turbines, req_data.battery_kwh, scenario_encoded]], columns=['num_solar_panels', 'num_wind_turbines', 'battery_kwh', 'scenario_encoded']); predictions = {target: round(model.predict(input_df)[0], 2) for target, model in ml_models.items()}; return jsonify({"status": "ML prediction successful", "predictions": predictions}), 200
@app.route('/chat', methods=['POST'])
def chat():
    if not azure_client: return jsonify({"error": "AI Advisor is disabled. API key may be missing or invalid. Check API server logs."}), 503
    user_query = request.json.get('query')
    if not user_query: return jsonify({"error": "Query is missing"}), 400
    try: intent_prompt = INTENT_PARSING_PROMPT.format(query=user_query); intent_response = azure_client.chat.completions.create(model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"), messages=[{"role": "user", "content": intent_prompt}], temperature=0.0, response_format={"type": "json_object"}); parsed_intent = json.loads(intent_response.choices[0].message.content); esg_weights = parsed_intent.get('esg_weights', {}); weight_sum = sum(esg_weights.values())
    except Exception as e: return jsonify({"error": "Could not parse request."}), 500
    if weight_sum > 0: parsed_intent['esg_weights'] = {k: v / weight_sum for k, v in esg_weights.items()}
    solution, _, _, _, _ = decision_engine.run_full_pipeline(**parsed_intent)
    if solution is None: ai_response = "Based on my interpretation of your request, our model could not find a feasible solution."
    else: context = {"user_query": user_query, "recommendation": {"Total Project Cost": f"€{solution['total_cost']:,.0f}", "Annual Savings": f"€{solution['annual_savings_eur']:,.0f}", "Payback Period": f"{solution['payback_period_years']:.1f} Years", "Self-Sufficiency": f"{solution['self_sufficiency_pct']:.1f}%", "Recommended Solar Panels": f"{solution['num_solar_panels']:,} units", "Recommended Wind Turbines": f"{solution['num_wind_turbines']:,} units", "Recommended Battery": f"{solution['battery_kwh']:,} kWh"}}; response_prompt = RESPONSE_GENERATION_PROMPT.format(context=json.dumps(context, indent=2)); final_response = azure_client.chat.completions.create(model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"), messages=[{"role": "user", "content": response_prompt}], temperature=0.4); ai_response = final_response.choices[0].message.content
    return jsonify({"response": ai_response})