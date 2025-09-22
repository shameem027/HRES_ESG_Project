# --- File: ui.py ---
import streamlit as st
import requests
import pandas as pd
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="HRES ESG Recommender",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Configuration ---
# The UI container will use the service name 'hres_api' to communicate
API_BASE_URL = "http://hres_api:8080"


# --- Helper Functions ---
def format_currency(value):
    return f"€{value:,.2f}"


def format_number(value):
    return f"{value:,.0f}"


# --- Sidebar ---
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("HRES ESG Recommender")
st.sidebar.markdown("A PhD prototype for sustainable energy decision support.")

# --- Main Application ---
st.title("💡 HRES ESG Recommender System")

tab1, tab2, tab3 = st.tabs(["📊 ESG Recommender", "⚡ ML Fast Predictor", "ℹ️ About"])

# ==============================================================================
# TAB 1: ESG Recommender
# ==============================================================================
with tab1:
    st.header("Find the Optimal Hybrid Renewable Energy System")
    st.markdown(
        "Define your facility's characteristics and prioritize your ESG goals. The system will run a full simulation "
        "and multi-criteria analysis to find the best-fit solution from thousands of possibilities."
    )

    with st.form("recommender_form"):
        col1, col2 = st.columns(2)
        with col1:
            scenario_name = st.selectbox(
                "Select Facility Type",
                ("Small_Office", "University_Campus", "Hospital", "Industrial_Facility", "Data_Center"),
                help="Choose the profile that best matches your facility's energy usage patterns."
            )
            annual_demand_kwh = st.number_input(
                "Annual Electricity Demand (kWh)",
                min_value=10000,
                value=250000,
                step=10000,
                help="Total kWh your facility consumes in a year."
            )
            user_grid_dependency_pct = st.slider(
                "Maximum Allowed Grid Dependency (%)",
                0, 100, 30,
                help="What percentage of your energy are you willing to import from the grid? 0% means fully off-grid."
            )

        with col2:
            st.markdown("**ESG & Cost Priorities**")
            cost_weight = st.slider("Cost Focus", 0.0, 1.0, 0.25, 0.05)
            env_weight = st.slider("Environmental Focus", 0.0, 1.0, 0.25, 0.05)
            social_weight = st.slider("Social Focus", 0.0, 1.0, 0.25, 0.05)
            gov_weight = st.slider("Governance Focus", 0.0, 1.0, 0.25, 0.05)

            # Normalize weights to sum to 1.0
            total_weight = cost_weight + env_weight + social_weight + gov_weight
            if total_weight > 0:
                cost_weight /= total_weight
                env_weight /= total_weight
                social_weight /= total_weight
                gov_weight /= total_weight

        submitted = st.form_submit_button("🚀 Find Best Solution", use_container_width=True)

    if submitted:
        payload = {
            "scenario_name": scenario_name,
            "annual_demand_kwh": annual_demand_kwh,
            "user_grid_dependency_pct": user_grid_dependency_pct,
            "esg_weights": {
                "cost": cost_weight,
                "environment": env_weight,
                "social": social_weight,
                "governance": gov_weight
            }
        }
        try:
            with st.spinner("Running simulations... This may take a moment."):
                response = requests.post(f"{API_BASE_URL}/recommend", json=payload)

            if response.status_code == 200:
                st.session_state.recommendation = response.json()
            elif response.status_code == 404:
                st.session_state.recommendation = None
                st.warning(f"**No Solution Found.** The API reported: *'{response.json().get('status', 'N/A')}'* "
                           "This usually means the constraints (like 0% grid dependency) are too strict. Try relaxing them.")
            else:
                st.session_state.recommendation = None
                st.error(f"An error occurred: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            st.session_state.recommendation = None
            st.error(f"Could not connect to the backend API. Please ensure it is running. Error: {e}")

    if 'recommendation' in st.session_state and st.session_state.recommendation:
        res = st.session_state.recommendation['recommendation']
        st.success(f"**Recommendation Found!** (Status: *{st.session_state.recommendation.get('status', 'OK')}*)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Initial Cost", format_currency(res['total_cost']))
        c2.metric("Payback Period", f"{res['payback_period_years']:.1f} Years")
        c3.metric("Annual Savings", format_currency(res['annual_savings_eur']))
        c4.metric("Self-Sufficiency", f"{res['self_sufficiency_pct']:.1f}%")

        st.subheader("Recommended System Configuration")
        config_col, kpi_col = st.columns(2)
        with config_col:
            st.markdown(f"""
            - **Solar Panels:** `{format_number(res['num_solar_panels'])}`
            - **Wind Turbines:** `{format_number(res['num_wind_turbines'])}`
            - **Battery Storage:** `{format_number(res['battery_kwh'])} kWh`
            """)
        with kpi_col:
            st.markdown(f"""
            - **Annual Generation:** `{format_number(res['annual_kwh_generated'])} kWh`
            - **CO2 Reduction:** `{format_number(res['env_co2_reduction_tons_yr'])} tons/yr`
            - **Energy Resilience:** `{res['soc_energy_resilience_hrs']:.1f} hours`
            """)

        with st.expander("Expand for Detailed Financial Breakdown"):
            constants = res.get('model_constants', {})
            st.json(constants, expanded=False)
            # You can also create a nice table from the constants dict
            # df_const = pd.DataFrame(list(constants.items()), columns=['Parameter', 'Value'])
            # st.table(df_const)

# ==============================================================================
# TAB 2: ML Fast Predictor
# ==============================================================================
with tab2:
    st.header("Get Instantaneous Performance Estimates")
    st.markdown(
        "Use our pre-trained Machine Learning models to get a quick estimate of a system's performance. "
        "This is useful for rapid prototyping before running a full simulation."
    )
    with st.form("ml_predictor_form"):
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            ml_scenario = st.selectbox("Facility Type", ("Small_Office", "Hospital", "University_Campus", "Data_Center",
                                                         "Industrial_Facility"), key="ml_scenario")
            num_solar = st.number_input("Number of Solar Panels", 0, 10000, 1000)
        with ml_col2:
            num_wind = st.number_input("Number of Wind Turbines", 0, 500, 50)
            battery_kwh = st.number_input("Battery Storage (kWh)", 0, 20000, 2000)

        ml_submitted = st.form_submit_button("⚡ Predict Performance", use_container_width=True)

    if ml_submitted:
        payload = {
            "scenario_name": ml_scenario,
            "num_solar_panels": num_solar,
            "num_wind_turbines": num_wind,
            "battery_kwh": battery_kwh
        }
        try:
            with st.spinner("Asking the ML model..."):
                response = requests.post(f"{API_BASE_URL}/predict_ml", json=payload)

            if response.status_code == 200:
                st.session_state.ml_prediction = response.json()['predictions']
            else:
                st.session_state.ml_prediction = None
                st.error(f"Prediction failed: {response.status_code} - {response.json().get('error', 'Unknown error')}")

        except requests.exceptions.RequestException as e:
            st.session_state.ml_prediction = None
            st.error(f"Could not connect to the backend API. Error: {e}")

    if 'ml_prediction' in st.session_state and st.session_state.ml_prediction:
        pred = st.session_state.ml_prediction
        st.success("Prediction Successful!")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Predicted Total Cost", format_currency(pred['total_cost']))
        p_col2.metric("Predicted Annual Savings", format_currency(pred['annual_savings_eur']))
        p_col3.metric("Predicted Self-Sufficiency", f"{pred['self_sufficiency_pct']:.1f}%")

# ==============================================================================
# TAB 3: About
# ==============================================================================
with tab3:
    st.header("About the Project")
    st.markdown("""
    This application is a prototype developed as part of a PhD research project by **Md Shameem Hossain**. 
    It demonstrates a full MLOps pipeline for a Hybrid Renewable Energy System (HRES) recommender that integrates 
    Environmental, Social, and Governance (ESG) criteria into its decision-making process.

    **Core Components:**
    - **Data Generation:** A sophisticated simulation engine generates a dataset of thousands of potential HRES configurations.
    - **Decision Engine (MCDA):** Uses Multi-Criteria Decision Analysis (MCDA) and ELECTRE-TRI methods to filter and rank solutions based on user preferences.
    - **ML Prediction:** Employs Machine Learning models for rapid performance estimation.
    - **Automation (Airflow):** An Apache Airflow pipeline automates the entire lifecycle of data generation, model training, and validation.
    - **Tracking (MLflow):** All experiments, models, and results are tracked using MLflow for reproducibility and governance.
    - **API (Flask):** A robust backend API serves the decision logic and ML models.
    - **UI (Streamlit):** This interactive web application provides the user interface.

    The entire stack is containerized using **Docker** for portability and ease of deployment.
    """)