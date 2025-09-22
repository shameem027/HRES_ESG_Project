# --- File: ui.py (Final Redesigned Version) ---
import streamlit as st
import requests
import pandas as pd
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="HRES ESG Recommender",
    page_icon="💡",
    layout="wide",
)

# --- API Configuration ---
API_BASE_URL = "http://hres_api:8080"  # For Docker


# API_BASE_URL = "http://localhost:8081" # For local dev

# --- Helper Functions ---
def format_currency(value): return f"€{value:,.0f}"


def format_number(value): return f"{value:,.0f}"


# --- SIDEBAR: Input Parameters ---
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.title("⚙️ HRES System Parameters")
    st.markdown("Define your facility and priorities. The model will find the optimal solution.")

    with st.form("recommender_form"):
        scenario_name = st.selectbox(
            "Facility Type",
            ("Small_Office", "University_Campus", "Hospital", "Industrial_Facility", "Data_Center")
        )
        annual_demand_kwh = st.number_input(
            "Annual Electricity Demand (kWh)",
            min_value=10000, value=250000, step=10000
        )
        user_grid_dependency_pct = st.slider(
            "Max. Grid Dependency (%)", 0, 100, 30
        )

        st.markdown("**ESG & Cost Priorities** (Weights will be normalized)")
        cost_weight = st.slider("Cost Focus", 0.0, 1.0, 0.25, 0.05)
        env_weight = st.slider("Environmental Focus", 0.0, 1.0, 0.25, 0.05)
        social_weight = st.slider("Social Focus", 0.0, 1.0, 0.25, 0.05)
        gov_weight = st.slider("Governance Focus", 0.0, 1.0, 0.25, 0.05)

        submitted = st.form_submit_button("🚀 Find Best Solution", use_container_width=True)

# Normalize weights
total_weight = cost_weight + env_weight + social_weight + gov_weight
if total_weight > 0:
    weights = {
        "cost": cost_weight / total_weight, "environment": env_weight / total_weight,
        "social": social_weight / total_weight, "governance": gov_weight / total_weight
    }
else:
    weights = {"cost": 0.25, "environment": 0.25, "social": 0.25, "governance": 0.25}

# --- MAIN WINDOW: TABS ---
st.title("💡 HRES ESG Recommender System")
tab_recommender, tab_predictor, tab_advisor, tab_about = st.tabs([
    "📊 ESG Recommender", "⚡ ML Fast Predictor", "🤖 AI Advisor", "ℹ️ About"
])

# --- TAB 1: ESG Recommender ---
with tab_recommender:
    st.header("Optimal Solution based on Full Simulation")
    if submitted:
        payload = {
            "scenario_name": scenario_name, "annual_demand_kwh": annual_demand_kwh,
            "user_grid_dependency_pct": user_grid_dependency_pct, "esg_weights": weights
        }
        try:
            with st.spinner("Running simulations... This may take a moment."):
                response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
            if response.status_code == 200:
                st.session_state.recommendation = response.json()
                st.success("✅ Recommendation Found!")
            elif response.status_code == 404:
                st.session_state.recommendation = None
                st.warning(
                    f"**No Solution Found.** API reported: *'{response.json().get('status', 'N/A')}'*. Try relaxing constraints.")
            else:
                st.session_state.recommendation = None
                st.error(f"An error occurred: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.session_state.recommendation = None
            st.error(f"Connection to backend API failed. Is it running? Error: {e}")

    if 'recommendation' in st.session_state and st.session_state.recommendation:
        res = st.session_state.recommendation['recommendation']

        # Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Initial Cost", format_currency(res['total_cost']))
        m2.metric("Annual Savings", format_currency(res['annual_savings_eur']))
        m3.metric("Payback Period", f"{res['payback_period_years']:.1f} Yrs")
        m4.metric("Self-Sufficiency", f"{res['self_sufficiency_pct']:.1f}%")

        # Visualizations
        st.subheader("Analysis & Visualizations")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Recommended System**")
            st.json({
                "Solar Panels": f"{format_number(res['num_solar_panels'])} units",
                "Wind Turbines": f"{format_number(res['num_wind_turbines'])} units",
                "Battery Storage": f"{format_number(res['battery_kwh'])} kWh"
            })
            st.markdown("**Key Performance Indicators**")
            st.json({
                "Annual Generation": f"{format_number(res['annual_kwh_generated'])} kWh",
                "CO2 Reduction": f"{format_number(res['env_co_reduction_tons_yr'])} tons/yr",
                "Energy Resilience": f"{res['soc_energy_resilience_hrs']:.1f} hours"
            })

        with c2:
            pareto_data = st.session_state.recommendation['intermediate_results']['pareto_front']
            if pareto_data:
                df_pareto = pd.DataFrame(pareto_data)
                st.markdown("**Pareto Front: Cost vs. Self-Sufficiency Trade-off**")
                st.scatter_chart(df_pareto, x='total_cost', y='self_sufficiency_pct', color='#ffaa00',
                                 size='annual_savings_eur')

        # Detailed Data
        with st.expander("View Full Solution Data Table"):
            df_res = pd.DataFrame([res])
            st.dataframe(df_res)

# --- TAB 2: ML Fast Predictor ---
with tab_predictor:
    st.header("Instantaneous Performance Estimate via Machine Learning")
    st.info("Configure a hypothetical system below to get a quick performance prediction from our trained ML models.")

    p_col1, p_col2, p_col3 = st.columns(3)
    ml_scenario = p_col1.selectbox("Facility Type", ("Small_Office", "Hospital", "University_Campus"),
                                   key="ml_scenario")
    num_solar = p_col2.number_input("Number of Solar Panels", 0, 10000, 1000)
    num_wind = p_col1.number_input("Number of Wind Turbines", 0, 500, 50)
    battery_kwh = p_col2.number_input("Battery Storage (kWh)", 0, 20000, 2000)

    if p_col3.button("⚡ Predict Performance", use_container_width=True):
        payload = {
            "scenario_name": ml_scenario, "num_solar_panels": num_solar,
            "num_wind_turbines": num_wind, "battery_kwh": battery_kwh
        }
        try:
            with st.spinner("Asking the ML model..."):
                response = requests.post(f"{API_BASE_URL}/predict_ml", json=payload)
            if response.status_code == 200:
                st.session_state.ml_prediction = response.json()['predictions']
            else:
                st.session_state.ml_prediction = None
                st.error(f"Prediction failed: {response.status_code} - {response.json().get('error', 'Unknown error')}")
        except requests.exceptions.RequestException:
            st.session_state.ml_prediction = None
            st.error("Connection to backend API failed.")

    if 'ml_prediction' in st.session_state and st.session_state.ml_prediction:
        pred = st.session_state.ml_prediction
        st.success("✅ Prediction Received!")
        pred_m1, pred_m2, pred_m3 = st.columns(3)
        pred_m1.metric("Predicted Total Cost", format_currency(pred['total_cost']))
        pred_m2.metric("Predicted Annual Savings", format_currency(pred['annual_savings_eur']))
        pred_m3.metric("Predicted Self-Sufficiency", f"{pred['self_sufficiency_pct']:.1f}%")

# --- TAB 3: AI Advisor ---
with tab_advisor:
    st.header("🤖 Chat with the AI Advisor")
    st.markdown("Use natural language to describe your needs. The advisor will configure the recommender for you.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g., I need a low cost solution for a small office with a green focus"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Consulting the AI Advisor..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/chat", json={"query": prompt})
                    if response.status_code == 200:
                        full_response = response.json()['response']
                    else:
                        full_response = f"Error: {response.json().get('error', 'Failed to get a response.')}"
                except requests.exceptions.RequestException:
                    full_response = "Error: Could not connect to the backend API."

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- TAB 4: About ---
with tab_about:
    st.header("About This Project")
    st.markdown("""
    This application is a prototype developed for the **MLOps and Project Management** course, demonstrating a full MLOps pipeline for a Hybrid Renewable Energy System (HRES) recommender.
    """)

    st.subheader("Business Value (15% of Grade)")
    st.markdown("""
    - **Problem:** Choosing the right renewable energy system is complex. Decision-makers must balance high upfront costs with long-term savings, while also considering increasingly important ESG (Environmental, Social, Governance) factors.
    - **Business Goal:** To create a decision-support tool that simplifies this complexity, reduces investment risk, and helps organizations achieve their sustainability targets.
    - **Value Proposition:** Our app provides instant, data-driven recommendations for optimal HRES configurations. It uniquely translates vague ESG goals (like "becoming greener") into quantifiable, optimized system designs, saving users time and consultancy fees.
    """)

    st.subheader("Technical Implementation & MLOps (60% of Grade)")
    st.markdown("""
    The entire system is a fully functioning demo built on a modern MLOps stack.
    - **Synthetic Data Generation:** A Python script (`HRES_Dataset_Generator.py`) simulates thousands of HRES configurations to create a rich, realistic dataset.
    - **ML Model:** A Random Forest model (`HRES_ML_Model.py`) is trained on this data to provide rapid predictions.
    - **Reproducibility & Tracking (MLflow):** Every experiment, including parameters, code versions, metrics, and model artifacts, is logged in **MLflow**. This ensures full reproducibility.
    - **Governance (MLflow Model Registry):** Trained models are versioned and registered in the MLflow Model Registry, providing a single source of truth for production-ready models.
    - **Automation (Airflow):** An **Apache Airflow** pipeline automates the entire MLOps lifecycle: it periodically regenerates the dataset, retrains the ML models, and runs validation tests, ensuring the system continuously improves.
    - **Containerization (Docker):** Every service (API, UI, Airflow, MLflow, Database) is containerized with **Docker**, guaranteeing a consistent environment from development to production.
    """)

    st.subheader("Presentation & Communication (10% of Grade)")
    st.markdown("""
    This UI serves as a key part of the presentation, providing a live, interactive demonstration of the project's capabilities. The pitch follows the story from business problem to technical solution and value creation.
    """)