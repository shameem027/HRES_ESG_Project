# --- File: ui.py (Definitive Final Version) ---
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# --- Page Configuration ---
st.set_page_config(page_title="HRES ESG Recommender", page_icon="💡", layout="wide")

# --- API Configuration ---
API_BASE_URL = "http://hres_api:8080"


# --- Helper Functions ---
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
        return None
    return None


# --- Main Application Logic ---
status = check_backend_status()

if not status or not status.get("decision_engine_loaded", False) or not status.get("ml_models_loaded", False):
    st.title("💡 HRES ESG Recommender System")
    st.info("""**System is Initializing...**
    The fully automated Airflow pipeline is generating data and training models. This one-time process may take 5-10 minutes on first startup.
    This page will automatically refresh. You can monitor live progress in the Airflow UI at [http://localhost:8080](http://localhost:8080) (user: airflow, pass: airflow).""",
            icon="⚙️")
    with st.spinner("Waiting for backend services to become ready..."):
        time.sleep(30)
    st.rerun()
else:
    # --- SIDEBAR: Input Parameters ---
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
            cost_weight = st.slider("Cost Focus", 0.0, 1.0, 0.25, 0.05)
            env_weight = st.slider("Environmental Focus", 0.0, 1.0, 0.25, 0.05)
            social_weight = st.slider("Social Focus", 0.0, 1.0, 0.25, 0.05)
            gov_weight = st.slider("Governance Focus", 0.0, 1.0, 0.25, 0.05)
            total_weight = cost_weight + env_weight + social_weight + gov_weight
            st.metric("Total Weight (Normalized to 1.0)", f"{1.0 if total_weight > 0 else 0.0:.2f}")
            submitted = st.form_submit_button("🚀 Find Best Solution", use_container_width=True)

    weights = {"cost": 0.25, "environment": 0.25, "social": 0.25, "governance": 0.25}
    if total_weight > 0:
        weights = {"cost": cost_weight / total_weight, "environment": env_weight / total_weight,
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
                with st.spinner("Analyzing thousands of configurations..."):
                    response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
                if response.status_code == 200:
                    st.session_state.recommendation = response.json()
                else:
                    st.session_state.recommendation = None
                    st.warning(
                        f"**No Solution Found.** API reported: *'{response.json().get('status', 'N/A')}'*. Please try relaxing the constraints in the sidebar.")
            except requests.exceptions.RequestException as e:
                st.session_state.recommendation = None
                st.error(f"Connection to the backend API failed. Error: {e}")

        if 'recommendation' in st.session_state and st.session_state.recommendation:
            res = st.session_state.recommendation['recommendation']
            st.success(f"**Optimal Solution Found!** (Status: *{st.session_state.recommendation.get('status')}*)")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Project Cost", format_currency(res['total_cost']))
            m2.metric("Annual Savings", format_currency(res['annual_savings_eur']))
            m3.metric("Payback Period", f"{res['payback_period_years']:.1f} Yrs")
            m4.metric("Self-Sufficiency", f"{res['self_sufficiency_pct']:.1f}%")
            st.markdown("---")

            v1, v2 = st.columns([1, 1])
            with v1:
                st.markdown("**Financial Summary**")
                df_financial = pd.DataFrame({'Metric': ["Total Project Cost", "Annual Maintenance",
                                                        "Annual Financing Cost", "Annual Savings",
                                                        "Payback Period (Years)"],
                                             'Value': [format_currency(res['total_cost']),
                                                       format_currency(res['annual_maintenance_cost_eur']),
                                                       format_currency(res['annual_financing_cost_eur']),
                                                       format_currency(res['annual_savings_eur']),
                                                       f"{res['payback_period_years']:.1f}"]})
                st.table(df_financial)
                st.markdown("**Project Cost Breakdown**")
                costs = {'Solar Panels': res['num_solar_panels'] * res['model_constants']['COST_PER_SOLAR_PANEL'],
                         'Wind Turbines': res['num_wind_turbines'] * res['model_constants']['COST_PER_WIND_TURBINE'],
                         'Battery': res['battery_kwh'] * res['model_constants']['COST_PER_BATTERY_KWH']}
                hardware_cost = sum(costs.values());
                costs['Installation & Overheads'] = res['total_cost'] - hardware_cost
                df_costs = pd.DataFrame(list(costs.items()), columns=['Category', 'Cost (€)']);
                fig_pie = px.pie(df_costs, values='Cost (€)', names='Category', title="Project Cost Distribution",
                                 hole=.3, color_discrete_sequence=px.colors.sequential.Tealgrn);
                st.plotly_chart(fig_pie, use_container_width=True)
            with v2:
                st.markdown("**ESG Score Radar Chart**")
                scores = res.get('normalized_scores', {});
                if scores:
                    df_scores = pd.DataFrame(dict(r=list(scores.values()), theta=list(scores.keys())));
                    fig_radar = go.Figure(
                        data=go.Scatterpolar(r=df_scores['r'], theta=df_scores['theta'], fill='toself',
                                             name='Normalized Score'));
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False);
                    st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("**Detailed ESG KPI Breakdown**")
            kpis = res.get('raw_kpis', {});
            e_kpis, s_kpis, g_kpis = st.columns(3)
            with e_kpis:
                with st.expander("🌳 Environmental KPIs", expanded=True): st.table(pd.DataFrame(
                    {'KPI': ['CO2 Reduction (tons/yr)', 'Land Use (sqm)', 'Water Savings (m3/yr)', 'Energy Waste (%)'],
                     'Value': [format_number(kpis.get('env_co2_reduction_tons_yr', 0)),
                               format_number(kpis.get('env_land_use_sqm', 0)),
                               format_number(kpis.get('env_water_savings_m3_yr', 0)),
                               f"{kpis.get('env_waste_factor_pct', 0):.1f}"]}))
            with s_kpis:
                with st.expander("🤝 Social KPIs", expanded=True): st.table(pd.DataFrame(
                    {'KPI': ['Local Jobs Created', 'Energy Resilience (hrs)', 'Grid Strain Reduction (%)'],
                     'Value': [f"{kpis.get('soc_local_jobs_fte', 0):.2f}",
                               f"{kpis.get('soc_energy_resilience_hrs', 0):.1f}",
                               f"{kpis.get('soc_grid_strain_reduction_pct', 0):.1f}"]}))
            with g_kpis:
                with st.expander("🏛️ Governance KPIs", expanded=True): st.table(pd.DataFrame(
                    {'KPI': ['Payback Plausibility (1-10)', 'Supply Chain Score (1-10)'],
                     'Value': [f"{kpis.get('gov_payback_plausibility_score', 0):.1f}",
                               f"{kpis.get('gov_supply_chain_transparency_score', 0):.1f}"]}))

            st.markdown("**Pareto Front: Feasible Solution Trade-offs**")
            pareto_data = st.session_state.recommendation.get('intermediate_results', {}).get('pareto_front', [])
            if pareto_data:
                df_pareto = pd.DataFrame(pareto_data);
                fig_scatter = px.scatter(df_pareto, x='total_cost', y='self_sufficiency_pct',
                                         title="Cost vs. Self-Sufficiency", color_discrete_sequence=['#ffaa00'],
                                         hover_data=['num_solar_panels', 'num_wind_turbines', 'battery_kwh'],
                                         labels={'total_cost': 'Total Cost (€)',
                                                 'self_sufficiency_pct': 'Self-Sufficiency (%)'});
                st.plotly_chart(fig_scatter, use_container_width=True)
        elif not submitted:
            st.info("Please configure your parameters in the sidebar and click 'Find Best Solution' to begin.")

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
            try:
                with st.spinner("Asking the ML model..."):
                    response = requests.post(f"{API_BASE_URL}/predict_ml", json=payload)
                if response.status_code == 200:
                    st.session_state.ml_prediction = response.json()['predictions']
                else:
                    st.session_state.ml_prediction = None; st.error(
                        f"Prediction failed: {response.status_code} - {response.json().get('error', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.session_state.ml_prediction = None; st.error(f"Connection to the backend API failed. Error: {e}")
        if 'ml_prediction' in st.session_state and st.session_state.ml_prediction:
            pred = st.session_state.ml_prediction
            st.success("✅ Prediction Received!");
            pred_m1, pred_m2, pred_m3 = st.columns(3);
            pred_m1.metric("Predicted Total Cost", format_currency(pred['total_cost']));
            pred_m2.metric("Predicted Annual Savings", format_currency(pred['annual_savings_eur']));
            pred_m3.metric("Predicted Self-Sufficiency", f"{pred['self_sufficiency_pct']:.1f}%")

    with tab_advisor:
        st.header("🤖 Chat with the AI Advisor")
        st.markdown("Use natural language to describe your needs.")
        if "messages" not in st.session_state: st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you configure your system today?"}]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        # This will be rendered inside the tab, but st.chat_input is now at the main level
        # which is the correct way to handle it.

    with tab_about:
        st.header("About This Project")
        st.markdown(
            "This application is a prototype developed for the **MLOps and Project Management** course, structured to meet the evaluation guidelines.")
        st.subheader("I. Business Value (15%)")
        st.markdown(
            "- **Business Problem:** Selecting an optimal Hybrid Renewable Energy System (HRES) is a high-stakes, complex decision...\n- **Business Goal:** To de-risk this investment...\n- **Customer Value:** Our \"BonsAI\" app empowers facility managers...")
        st.subheader("II. MLOps Workflow & Technical Implementation (60%)")
        st.markdown(
            "- **Synthetic Data Generation:** A Python script (`HRES_Dataset_Generator.py`) simulates thousands of HRES configurations...\n- **ML/AI Core:**...\n- **Reproducibility & Tracking (MLflow):** Every experiment is tracked in **MLflow**...\n- **Governance (MLflow Model Registry):**...\n- **Automation (Airflow):** An **Apache Airflow** pipeline...\n- **Deployment (Docker):** Every service is containerized with **Docker**...")
        st.subheader("III. Fulfillment of Mini-Project Requirements")
        st.markdown(
            "- **[✔] Business Scope:** The problem, goal, and customer value are clearly articulated...\n- **[✔] ML/AI Scope:**...\n- **[✔] MLOps Scope (MLflow):**...\n- **[✔] Automation (Airflow Bonus):**...\n- **[✔] Production Scenario (Bonus):**...")

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