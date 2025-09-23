# --- File: ui.py (Definitive, Simplified, Final Version) ---
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="HRES ESG Recommender", page_icon="💡", layout="wide")
API_BASE_URL = "http://hres_api:8080"
def format_currency(value): return f"€{value:,.0f}"
def format_number(value): return f"{value:,.0f}"

with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.title("⚙️ HRES System Parameters")
    st.markdown("Define your facility and priorities to find the optimal HRES solution.")
    with st.form("recommender_form"):
        scenario_name = st.selectbox("Facility Type", ("Small_Office", "University_Campus", "Hospital", "Industrial_Facility", "Data_Center"))
        annual_demand_kwh = st.number_input("Annual Electricity Demand (kWh)", min_value=10000, value=250000, step=10000)
        user_grid_dependency_pct = st.slider("Max. Grid Dependency (%)", 0, 100, 30)
        st.markdown("**ESG & Cost Priorities**")
        cost_weight = st.slider("Cost Focus", 0.0, 1.0, 0.25, 0.05)
        env_weight = st.slider("Environmental Focus", 0.0, 1.0, 0.25, 0.05)
        social_weight = st.slider("Social Focus", 0.0, 1.0, 0.25, 0.05)
        gov_weight = st.slider("Governance Focus", 0.0, 1.0, 0.25, 0.05)
        total_weight = cost_weight + env_weight + social_weight + gov_weight
        st.metric("Total Weight (Normalized)", f"{1.0:.2f}")
        submitted = st.form_submit_button("🚀 Find Best Solution", use_container_width=True)

weights = {"cost": 0.25, "environment": 0.25, "social": 0.25, "governance": 0.25}
if total_weight > 0: weights = {"cost": cost_weight / total_weight, "environment": env_weight / total_weight, "social": social_weight / total_weight, "governance": gov_weight / total_weight}

st.title("💡 HRES ESG Recommender System")
tab_recommender, tab_predictor, tab_advisor, tab_about = st.tabs(["📊 ESG Recommender", "⚡ ML Fast Predictor", "🤖 AI Advisor", "ℹ️ About"])

with tab_recommender:
    st.header("Optimal Solution Dashboard")
    if submitted:
        payload = {"scenario_name": scenario_name, "annual_demand_kwh": annual_demand_kwh, "user_grid_dependency_pct": user_grid_dependency_pct, "esg_weights": weights}
        try:
            with st.spinner("Analyzing configurations..."): response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
            if response.status_code == 200: st.session_state.recommendation = response.json()
            else: st.session_state.recommendation = None; st.warning(f"**No Solution Found.** API reported: *'{response.json().get('status', 'N/A')}'*.")
        except requests.exceptions.RequestException as e: st.session_state.recommendation = None; st.error(f"Connection to backend API failed. Please ensure the Docker stack is running. Error: {e}")
    if 'recommendation' in st.session_state and st.session_state.recommendation:
        res = st.session_state.recommendation['recommendation']
        st.success(f"**Optimal Solution Found!**"); m1, m2, m3, m4 = st.columns(4); m1.metric("Total Project Cost", format_currency(res['total_cost'])); m2.metric("Annual Savings", format_currency(res['annual_savings_eur'])); m3.metric("Payback Period", f"{res['payback_period_years']:.1f} Yrs"); m4.metric("Self-Sufficiency", f"{res['self_sufficiency_pct']:.1f}%")
        # ... (All dashboard visualizations remain the same)

with tab_predictor:
    st.header("Instantaneous Performance Estimate via Machine Learning")
    st.warning("️️️⚠️ **Prerequisite:** This feature requires the ML models to be trained. If this is the first time running the system, please go to the **Airflow UI** (`http://localhost:8080`), un-pause the `HRES_PhD_Automation_Pipeline` DAG, and trigger a manual run.")
    # ... (ML Predictor form remains the same)

with tab_advisor:
    st.header("🤖 Chat with the AI Advisor")
    st.markdown("Use natural language to describe your needs.")
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "How can I help you configure your system today?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask the AI Advisor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Consulting..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/chat", json={"query": prompt})
                    if response.status_code == 200: assistant_response = response.json()['response']
                    else: assistant_response = f"Error: {response.json().get('error', 'Failed to get a response.')}"
                except requests.exceptions.RequestException as e: assistant_response = f"Error: Could not connect to the API. {e}"
            message_placeholder.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

with tab_about:
    # ... (About tab content is correct and remains the same)
    pass