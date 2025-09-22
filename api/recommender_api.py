# --- File: api/recommender_api.py (Final Version with Enhanced Prompts) ---
import os
import pandas as pd
from flask import Flask, request, jsonify
import logging
import json
from openai import AzureOpenAI
import mlflow
import mlflow.sklearn
from pydantic import BaseModel, ValidationError, Field

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic Models for Input Validation ---
class EsgWeights(BaseModel):  # ... as before
    environment: float = Field(..., ge=0, le=1)
    social: float = Field(..., ge=0, le=1)
    governance: float = Field(..., ge=0, le=1)
    cost: float = Field(..., ge=0, le=1)


class RecommendRequest(BaseModel):  # ... as before
    scenario_name: str
    annual_demand_kwh: int = Field(..., gt=0)
    user_grid_dependency_pct: int = Field(..., ge=0, le=100)
    esg_weights: EsgWeights


class MlPredictRequest(BaseModel):  # ... as before
    scenario_name: str
    num_solar_panels: int = Field(..., ge=0)
    num_wind_turbines: int = Field(..., ge=0)
    battery_kwh: int = Field(..., ge=0)


# --- Global Variables ---
decision_engine = None
ml_models = {}
scenario_encoders = {}
azure_client = None

# --- EXTENSIVE PROMPT ENGINEERING ---
INTENT_PARSING_PROMPT = """
You are a hyper-prescriptive AI data extractor. Your ONLY task is to parse a user's query into a valid JSON object.
STRICTLY adhere to the following rules:
1.  Output ONLY a single JSON object. No introductory text, no explanations, no apologies.
2.  The JSON object MUST contain ALL the following keys: `scenario_name`, `annual_demand_kwh`, `user_grid_dependency_pct`, `esg_weights`.
3.  `scenario_name`: MUST be one of ["Small_Office", "University_Campus", "Hospital", "Industrial_Facility", "Data_Center"]. Infer the closest match.
4.  `annual_demand_kwh`: MUST be an integer. Use these defaults if not specified: Small_Office: 250000, Hospital: 1500000, University_Campus: 3000000, Industrial_Facility: 5000000, Data_Center: 10000000.
5.  `user_grid_dependency_pct`: MUST be an integer between 0-100. Infer from keywords: "off-grid"/"independent" -> 0; "high resilience"/"minimal grid" -> 10; "balanced" -> 30. Default to 30.
6.  `esg_weights`: MUST be an object with keys "environment", "social", "governance", "cost". The sum of their values MUST be 1.0.
    - "low cost"/"cheap"/"economical": high "cost" weight (e.g., 0.7).
    - "green"/"eco-friendly"/"sustainable": high "environment" weight (e.g., 0.7).
    - "community"/"social benefit": high "social" weight (e.g., 0.7).
    - If priorities are mixed (e.g., "cheap and green"), distribute the weights (e.g., cost: 0.45, environment: 0.45, social: 0.05, governance: 0.05).
    - If no priority is given, default to a balanced split (0.25 each).

User Query: "{query}"

JSON Output:
"""

RESPONSE_GENERATION_PROMPT = """
You are an expert AI energy consultant. Your task is to provide a professional, data-driven summary based on a quantitative model's recommendation.
STRICTLY adhere to the following rules:
1.  Acknowledge the user's request from the `user_query` in the context.
2.  Provide a concise, helpful summary in one paragraph.
3.  **IMPORTANT**: Present the key quantitative results in a well-formatted Markdown table. The table should have two columns: "Metric" and "Value".
4.  Do not invent any data not present in the `recommendation` context.
5.  Maintain a professional and encouraging tone.

CONTEXT:
{context}

AI Consultant Response (with Markdown table):
"""


# --- End Prompts ---

def load_resources():  # ... as before
    global decision_engine, ml_models, scenario_encoders, azure_client
    try:
        from src.MCDA_model import HRES_Decision_Engine
        DATA_PATH = "/app/src/HRES_Dataset.csv"
        hres_df = pd.read_csv(DATA_PATH)
        decision_engine = HRES_Decision_Engine(hres_df)
        logger.info(f"✅ HRES Decision Engine initialized.")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Could not initialize Decision Engine. Error: {e}", exc_info=True)
    try:
        from src.HRES_ML_Model import HRESMLPredictor
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        for target in ['total_cost', 'self_sufficiency_pct', 'annual_savings_eur']:
            model, encoder = HRESMLPredictor.load_latest_model(target, model_name_suffix="_V1")
            if model and encoder: ml_models[target] = model; scenario_encoders[target] = encoder
        logger.info(f"✅ Loaded {len(ml_models)} ML models.")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Could not load ML models. Error: {e}", exc_info=True)
    try:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            azure_client = AzureOpenAI(api_version=os.getenv("OPENAI_API_VERSION"),
                                       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                       api_key=os.getenv("AZURE_OPENAI_API_KEY"))
            logger.info("✅ Azure OpenAI client initialized.")
        else:
            logger.warning("⚠️ AZURE_OPENAI_API_KEY not set. Chat functionality disabled.")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize Azure OpenAI client. Error: {e}.")


with app.app_context(): load_resources()


@app.route('/health', methods=['GET'])
def health_check():  # ... as before
    return jsonify({"status": "healthy", "decision_engine_loaded": decision_engine is not None,
                    "llm_client_loaded": azure_client is not None, "ml_models_loaded": len(ml_models) > 0})


@app.route('/recommend', methods=['POST'])
def recommend():  # ... as before
    if not decision_engine: return jsonify({"error": "Decision Engine not operational."}), 503
    try:
        req_data = RecommendRequest.model_validate(request.get_json())
        best_solution, status, feasible_df, sorted_df, pareto_df = decision_engine.run_full_pipeline(
            req_data.scenario_name, req_data.annual_demand_kwh, req_data.user_grid_dependency_pct,
            req_data.esg_weights.model_dump())
        if best_solution is not None:
            return jsonify({"status": status, "recommendation": best_solution.to_dict(), "intermediate_results": {
                "pareto_front": pareto_df.to_dict(orient='records') if pareto_df is not None else []}}), 200
        else:
            return jsonify({"status": status, "recommendation": None}), 404
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 422
    except Exception as e:
        logger.error(f"Error in /recommend: {e}", exc_info=True); return jsonify(
            {"error": "Internal server error"}), 500


@app.route('/predict_ml', methods=['POST'])
def predict_ml():  # ... as before
    if not ml_models: return jsonify({"error": "ML models not loaded."}), 503
    try:
        req_data = MlPredictRequest.model_validate(request.get_json())
        encoder = scenario_encoders.get('total_cost')
        if not encoder or req_data.scenario_name not in encoder: return jsonify(
            {"error": "Scenario not recognized"}), 400
        scenario_encoded = encoder[req_data.scenario_name]
        input_df = pd.DataFrame(
            [[req_data.num_solar_panels, req_data.num_wind_turbines, req_data.battery_kwh, scenario_encoded]],
            columns=['num_solar_panels', 'num_wind_turbines', 'battery_kwh', 'scenario_encoded'])
        predictions = {target: round(model.predict(input_df)[0], 2) for target, model in ml_models.items()}
        return jsonify({"status": "ML prediction successful", "predictions": predictions}), 200
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 422
    except Exception as e:
        logger.error(f"Error in /predict_ml: {e}", exc_info=True); return jsonify(
            {"error": "Internal server error"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    if not azure_client: return jsonify({"error": "AI Advisor is disabled. Check API server logs."}), 503
    user_query = request.json.get('query')
    if not user_query: return jsonify({"error": "Query is missing"}), 400
    try:
        intent_prompt = INTENT_PARSING_PROMPT.format(query=user_query)
        intent_response = azure_client.chat.completions.create(model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
                                                               messages=[{"role": "user", "content": intent_prompt}],
                                                               temperature=0.0, response_format={"type": "json_object"})
        parsed_intent = json.loads(intent_response.choices[0].message.content)

        solution, _, _, _, _ = decision_engine.run_full_pipeline(**parsed_intent)

        if solution is None:
            ai_response = "Based on my interpretation of your request, our model could not find a feasible solution. This usually means the goals are too ambitious. Please try relaxing your constraints (e.g., allow more grid dependency)."
        else:
            context = {
                "user_query": user_query,
                "recommendation": {
                    "Total Project Cost": f"€{solution['total_cost']:,.0f}",
                    "Annual Savings": f"€{solution['annual_savings_eur']:,.0f}",
                    "Payback Period": f"{solution['payback_period_years']:.1f} Years",
                    "Self-Sufficiency": f"{solution['self_sufficiency_pct']:.1f}%",
                    "Recommended Solar Panels": f"{solution['num_solar_panels']:,} units",
                    "Recommended Wind Turbines": f"{solution['num_wind_turbines']:,} units",
                    "Recommended Battery": f"{solution['battery_kwh']:,} kWh"
                }
            }
            response_prompt = RESPONSE_GENERATION_PROMPT.format(context=json.dumps(context, indent=2))
            final_response = azure_client.chat.completions.create(model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
                                                                  messages=[
                                                                      {"role": "user", "content": response_prompt}],
                                                                  temperature=0.4)
            ai_response = final_response.choices[0].message.content
        return jsonify({"response": ai_response})
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while consulting the AI Advisor."}), 500