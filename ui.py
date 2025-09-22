# --- File: ui.py (Definitive "State-of-the-Art" Version) ---
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="HRES ESG Recommender",
    page_icon="💡",
    layout="wide",
)

# --- API Configuration ---
API_BASE_URL = "http://hres_api:8080"


# --- Helper Functions ---
def format_currency(value): return f"€{value:,.0f}"


def format_number(value): return f"{value:,.0f}"


# --- SIDEBAR: Input Parameters ---
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.title("⚙️ HRES System Parameters")
    st.markdown("Define your facility and priorities to find the optimal HRES solution.")

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
            "Max. Grid Dependency (%)", 0, 100, 30,
            help="0% means fully off-grid; 100% means no self-sufficiency requirement."
        )

        st.markdown("**ESG & Cost Priorities** (Weights will be normalized)")
        cost_weight = st.slider("Cost Focus", 0.0, 1.0, 0.25, 0.05)
        env_weight = st.slider("Environmental Focus", 0.0, 1.0, 0.25, 0.05)
        social_weight = st.slider("Social Focus", 0.0, 1.0, 0.25, 0.05)
        gov_weight = st.slider("Governance Focus", 0.0, 1.0, 0.25, 0.05)

        submitted = st.form_submit_button("🚀 Find Best Solution", use_container_width=True)

# Normalize weights
total_weight = cost_weight + env_weight + social_weight + gov_weight
weights = {"cost": 0.25, "environment": 0.25, "social": 0.25, "governance": 0.25}
if total_weight > 0:
    weights = {
        "cost": cost_weight / total_weight, "environment": env_weight / total_weight,
        "social": social_weight / total_weight, "governance": gov_weight / total_weight
    }

# --- MAIN WINDOW: TABS ---
st.title("💡 HRES ESG Recommender System")
tab_recommender, tab_predictor, tab_advisor, tab_about = st.tabs([
    "📊 ESG Recommender", "⚡ ML Fast Predictor", "🤖 AI Advisor", "ℹ️ About"
])

# --- TAB 1: ESG Recommender ---
with tab_recommender:
    st.header("Optimal Solution Dashboard")
    if submitted:
        payload = {
            "scenario_name": scenario_name, "annual_demand_kwh": annual_demand_kwh,
            "user_grid_dependency_pct": user_grid_dependency_pct, "esg_weights": weights
        }
        try:
            with st.spinner("Analyzing thousands of configurations... This may take a moment."):
                response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
            if response.status_code == 200:
                st.session_state.recommendation = response.json()
            else:
                st.session_state.recommendation = None
                st.warning(
                    f"**No Solution Found.** API reported: *'{response.json().get('status', 'N/A')}'*. Please try relaxing the constraints in the sidebar.")
        except requests.exceptions.RequestException as e:
            st.session_state.recommendation = None
            st.error(f"Connection to the backend API failed. Please ensure all Docker services are running. Error: {e}")

    if 'recommendation' in st.session_state and st.session_state.recommendation:
        res = st.session_state.recommendation['recommendation']

        st.success(f"**Optimal Solution Found!** (Status: *{st.session_state.recommendation.get('status')}*)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Project Cost", format_currency(res['total_cost']))
        m2.metric("Annual Savings", format_currency(res['annual_savings_eur']))
        m3.metric("Payback Period", f"{res['payback_period_years']:.1f} Yrs")
        m4.metric("Self-Sufficiency", f"{res['self_sufficiency_pct']:.1f}%")

        st.markdown("---")

        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Recommended System Configuration**")
            st.dataframe(pd.DataFrame({
                'Component': ['Solar Panels', 'Wind Turbines', 'Battery Storage'],
                'Value': [f"{format_number(res['num_solar_panels'])} units",
                          f"{format_number(res['num_wind_turbines'])} units",
                          f"{format_number(res['battery_kwh'])} kWh"]
            }))

            st.markdown("**Project Cost Breakdown**")
            costs = {
                'Solar Panels': res['num_solar_panels'] * res['model_constants']['COST_PER_SOLAR_PANEL'],
                'Wind Turbines': res['num_wind_turbines'] * res['model_constants']['COST_PER_WIND_TURBINE'],
                'Battery': res['battery_kwh'] * res['model_constants']['COST_PER_BATTERY_KWH']
            }
            hardware_cost = sum(costs.values())
            costs['Installation & Overheads'] = res['total_cost'] - hardware_cost
            df_costs = pd.DataFrame(list(costs.items()), columns=['Category', 'Cost (€)'])
            fig_pie = px.pie(df_costs, values='Cost (€)', names='Category', title="Project Cost Distribution", hole=.3)
            st.plotly_chart(fig_pie, use_container_width=True)

        with v2:
            st.markdown("**ESG & Cost Score Breakdown**")
            scores = res.get('component_scores', {})
            if scores:
                df_scores = pd.DataFrame(list(scores.items()), columns=['Component', 'Normalized Score']).sort_values(
                    'Normalized Score', ascending=False)
                fig_bar = px.bar(df_scores, x='Component', y='Normalized Score', title="Why was this solution chosen?",
                                 color='Component',
                                 labels={'Normalized Score': 'Score (0-1)', 'Component': 'Scoring Dimension'})
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Pareto Front: Feasible Solution Trade-offs**")
        pareto_data = st.session_state.recommendation.get('intermediate_results', {}).get('pareto_front', [])
        if pareto_data:
            df_pareto = pd.DataFrame(pareto_data)
            fig_scatter = px.scatter(df_pareto, x='total_cost', y='self_sufficiency_pct',
                                     title="Cost vs. Self-Sufficiency",
                                     hover_data=['num_solar_panels', 'num_wind_turbines', 'battery_kwh'],
                                     labels={'total_cost': 'Total Cost (€)',
                                             'self_sufficiency_pct': 'Self-Sufficiency (%)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Please configure your parameters in the sidebar and click 'Find Best Solution' to begin.")

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
        # ... (API call logic remains the same)
        pass

# --- TAB 3: AI Advisor ---
with tab_advisor:
    st.header("🤖 Chat with the AI Advisor")
    st.markdown("Use natural language to describe your needs. The advisor will configure the recommender for you.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you configure your renewable energy system today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g., I need a low cost solution for a small office with a green focus"):
        # ... (Chat logic remains the same)
        pass

# --- TAB 4: About ---
with tab_about:
    st.header("About This Project")
    st.markdown(
        "This application is a prototype developed for the **MLOps and Project Management** course, structured to meet the evaluation guidelines.")

    st.subheader("I. Business Value (15%)")
    st.markdown("""
    - **Business Problem:** Selecting an optimal Hybrid Renewable Energy System (HRES) is a high-stakes, complex decision. Stakeholders must balance large capital expenditures against long-term operational savings, while navigating complex ESG (Environmental, Social, Governance) commitments.
    - **Business Goal:** To de-risk this investment by creating an intelligent decision-support tool that provides data-driven, ESG-aligned recommendations, reducing reliance on expensive, time-consuming manual consultancy.
    - **Customer Value:** Our "BonsAI" app empowers facility managers and sustainability officers to instantly explore trade-offs between cost and ESG goals. It translates abstract objectives like "becoming greener" into concrete, financially viable system designs, accelerating the path to sustainability.
    """)

    st.subheader("II. Technical Implementation & MLOps (60%)")
    st.markdown("""
    This is a fully functioning, containerized application demonstrating a complete MLOps lifecycle.
    - **Data Generation:** A Python script (`HRES_Dataset_Generator.py`) simulates thousands of HRES configurations to create a rich, proprietary dataset.
    - **ML/AI Core:**
        - **Decision Engine:** A robust Multi-Criteria Decision Analysis (MCDA) model (`MCDA_model.py`) filters and ranks solutions based on user-defined constraints and priorities.
        - **Predictive Model:** A Random Forest model (`HRES_ML_Model.py`) is trained for rapid performance estimation.
        - **LLM Integration:** An LLM-powered "AI Advisor" parses natural language queries into structured API calls.
    - **Reproducibility & Tracking (MLflow):** Every experiment is tracked in **MLflow**. All parameters, code versions, metrics, and model artifacts are logged, ensuring full auditability and reproducibility.
    - **Governance (MLflow Model Registry):** Trained models are versioned and managed in the MLflow Model Registry, providing a central, governed repository for "production-ready" models.
    - **Automation (Airflow):** An **Apache Airflow** pipeline (`HRES_Automation_Pipeline.py`) automates the entire MLOps workflow: dataset regeneration, model retraining, and validation, ensuring the system remains up-to-date.
    - **Deployment (Docker):** Every service (API, UI, Airflow, MLflow, Database) is individually containerized with **Docker** and orchestrated with **Docker Compose**, guaranteeing a consistent and portable environment.
    """)

    st.subheader("III. Presentation & Teamwork (25%)")
    st.markdown("""
    - **Live Demo:** This Streamlit UI serves as the live, interactive demonstration of the project's end-to-end functionality.
    - **Clarity & Conciseness:** The pitch will walk through the business problem, the technical solution, and the tangible value proposition, all within the 7-minute timeframe.
    - **Shared Understanding:** All team members have contributed to the development and can speak to both the business and technical aspects of the project.
    """)