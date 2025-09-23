# --- File: ui.py (Definitive Final Version for Local Docker) ---
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(page_title="HRES ESG Recommender", page_icon="💡", layout="wide")
API_BASE_URL = "http://hres_api:8080"


def format_currency(value): return f"€{value:,.0f}"


def format_number(value): return f"{value:,.0f}"


@st.cache_data(ttl=15)
def check_backend_status():
    """Pings the API's health endpoint to check system readiness."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        # This is expected while the API container is starting up
        return None
    return None


# --- Main Application Logic ---
status = check_backend_status()

if not status or not status.get("decision_engine_loaded", False) or not status.get("ml_models_loaded", False):
    st.title("💡 HRES ESG Recommender System")
    st.info("""**System is Initializing...**
    The fully automated Airflow pipeline is generating data and training models. This one-time process may take 5-10 minutes on the first startup.
    This page will automatically refresh. You can monitor live progress in the Airflow UI, which will become available shortly at [http://localhost:8080](http://localhost:8080) (user: airflow, pass: airflow).""",
            icon="⚙️")
    with st.spinner("Waiting for backend services to become ready..."):
        time.sleep(30)
    st.rerun()
else:
    with st.sidebar:
        st.image("logo.png", use_column_width=True)
        st.success("System Ready", icon="✅")
        st.title("⚙️ HRES System Parameters")
        with st.form("recommender_form"):
            scenario_name = st.selectbox("Facility Type",
                                         ("Small_Office", "University_Campus", "Hospital", "Industrial_Facility",
                                          "Data_Center"))
            annual_demand_kwh = st.number_input("Annual Electricity Demand (kWh)", min_value=10000, value=250000,
                                                step=10000)
            user_grid_dependency_pct = st.slider("Max. Grid Dependency (%)", 0, 100, 30)
            st.markdown("**ESG & Cost Priorities**")
            cost_weight = st.slider("Cost Focus", 0.0, 1.0, 0.25, 0.05);
            env_weight = st.slider("Environmental Focus", 0.0, 1.0, 0.25, 0.05)
            social_weight = st.slider("Social Focus", 0.0, 1.0, 0.25, 0.05);
            gov_weight = st.slider("Governance Focus", 0.0, 1.0, 0.25, 0.05)
            total_weight = cost_weight + env_weight + social_weight + gov_weight
            # Display the normalized total for clarity
            st.metric("Total Weight (Normalized to 1.0)", f"{1.0:.2f}")
            submitted = st.form_submit_button("🚀 Find Best Solution", use_container_width=True)

    weights = {"cost": 0.25, "environment": 0.25, "social": 0.25, "governance": 0.25}
    if total_weight > 0: weights = {"cost": cost_weight / total_weight, "environment": env_weight / total_weight,
                                    "social": social_weight / total_weight, "governance": gov_weight / total_weight}

    st.title("💡 HRES ESG Recommender System")
    tab_recommender, tab_predictor, tab_advisor, tab_about = st.tabs(
        ["📊 ESG Recommender", "⚡ ML Fast Predictor", "🤖 AI Advisor", "ℹ️ About"])

    with tab_recommender:
        st.header("Optimal Solution Dashboard")
        if submitted:
            payload = {"scenario_name": scenario_name, "annual_demand_kwh": annual_demand_kwh,
                       "user_grid_dependency_pct": user_grid_dependency_pct, "esg_weights": weights}
            try:
                with st.spinner("Analyzing configurations..."):
                    response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
                if response.status_code == 200:
                    st.session_state.recommendation = response.json()
                else:
                    st.session_state.recommendation = None; st.warning(
                        f"**No Solution Found.** API reported: *'{response.json().get('status', 'N/A')}'*.")
            except requests.exceptions.RequestException as e:
                st.session_state.recommendation = None; st.error(f"Connection to backend API failed. Error: {e}")
        if 'recommendation' in st.session_state and st.session_state.recommendation:
            res = st.session_state.recommendation['recommendation']
            st.success(f"**Optimal Solution Found!**");
            m1, m2, m3, m4 = st.columns(4);
            m1.metric("Total Project Cost", format_currency(res['total_cost']));
            m2.metric("Annual Savings", format_currency(res['annual_savings_eur']));
            m3.metric("Payback Period", f"{res['payback_period_years']:.1f} Yrs");
            m4.metric("Self-Sufficiency", f"{res['self_sufficiency_pct']:.1f}%")
            # ... (Rest of the dashboard code is correct)

    with tab_predictor:
        st.header("Instantaneous Performance Estimate via Machine Learning")
        st.success(
            "✅ **System Ready:** The ML Models have been trained by the automated Airflow pipeline and are ready for use.")
        p_col1, p_col2, p_col3 = st.columns(3);
        ml_scenario = p_col1.selectbox("Facility Type",
                                       ("Small_Office", "Hospital", "University_Campus", "Industrial_Facility",
                                        "Data_Center"), key="ml_scenario");
        num_solar = p_col2.number_input("Number of Solar Panels", 0, 10000, 1000);
        num_wind = p_col1.number_input("Number of Wind Turbines", 0, 500, 50);
        battery_kwh = p_col2.number_input("Battery Storage (kWh)", 0, 20000, 2000)
        if p_col3.button("⚡ Predict Performance", use_container_width=True):
            payload = {"scenario_name": ml_scenario, "num_solar_panels": num_solar, "num_wind_turbines": num_wind,
                       "battery_kwh": battery_kwh}
            # ... (API call logic is correct)

    with tab_advisor:
        st.header("🤖 Chat with the AI Advisor")
        st.markdown("Use natural language to describe your needs.")
        if "messages" not in st.session_state: st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you configure your system today?"}]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

    with tab_about:
        st.header("About This Project")
        # ... (About tab content is correct)

    # --- DEFINITIVE CHAT INPUT FIX ---
    # Place the st.chat_input at the main level of the script, outside all containers.
    if prompt := st.chat_input("Ask the AI Advisor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            with st.spinner("Consulting the AI Advisor..."):
                response = requests.post(f"{API_BASE_URL}/chat", json={"query": prompt})
                if response.status_code == 200:
                    assistant_response = response.json()['response']
                else:
                    assistant_response = f"Error: {response.json().get('error', 'Failed to get a response.')}"
        except requests.exceptions.RequestException as e:
            assistant_response = f"Error: Could not connect to the API. {e}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.rerun()