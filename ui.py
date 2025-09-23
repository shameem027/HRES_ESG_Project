import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# API Base URL
API_BASE_URL = "http://hres_api:8080"

def format_currency(value):
    return f"€{value:,.0f}"

def format_number(value):
    return f"{value:,.0f}"

# --- ESG Weights Model ---
class EsgWeights(st.runtime.model):
    environment: float = 0.25
    social: float = 0.25
    governance: float = 0.25
    cost: float = 0.25

# --- Recommend Request Model ---
class RecommendRequest(st.runtime.model):
    scenario_name: str
    annual_demand_kwh: int = 250000
    user_grid_dependency_pct: int = 30
    esg_weights: EsgWeights = EsgWeights()

# --- UI Layout ---
st.set_page_config(page_title="HRES ESG Recommender", page_icon="💡", layout="wide")

with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.success("System Ready", icon="✅")
    st.title("⚙️ HRES System Parameters")

    with st.form("recommender_form"):
        scenario_name = st.selectbox("Facility Type", ("Small_Office", "University_Campus", "Hospital", "Industrial_Facility", "Data_Center"))
        annual_demand_kwh = st.number_input("Annual Electricity Demand (kWh)", min_value=10000, value=250000, step=10000)
        user_grid_dependency_pct = st.slider("Max. Grid Dependency (%)", 0, 100, 30)

        st.markdown("**ESG & Cost Priorities**")
        esg_weights = EsgWeights()
        esg_weights.cost = st.slider("Cost Focus", 0.0, 1.0, 0.25, 0.05)
        esg_weights.environment = st.slider("Environmental Focus", 0.0, 1.0, 0.25, 0.05)
        esg_weights.social = st.slider("Social Focus", 0.0, 1.0, 0.25, 0.05)
        esg_weights.governance = st.slider("Governance Focus", 0.0, 1.0, 0.25, 0.05)

        total_weight = esg_weights.cost + esg_weights.environment + esg_weights.social + esg_weights.governance
        st.metric("Total Weight (Normalized to 1.0)", f"{total_weight:.2f}")

        submitted = st.form_submit_button("🚀 Find Best Solution", use_container_width=True)

st.title("💡 HRES ESG Recommender System")

tab_recommender, tab_predictor, tab_advisor, tab_about = st.tabs(["📊 ESG Recommender", "⚡ ML Fast Predictor", "🤖 AI Advisor", "ℹ️ About"])

with tab_recommender:
    st.header("Optimal Solution Dashboard")
    if submitted:
        request_data = RecommendRequest(scenario_name=scenario_name, annual_demand_kwh=annual_demand_kwh, user_grid_dependency_pct=user_grid_dependency_pct, esg_weights=esg_weights)
        try:
            response = requests.post(f"{API_BASE_URL}/recommend", json=request_data.model_dump())
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            recommendation = response.json()
            st.success(f"**Optimal Solution Found!**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Project Cost", format_currency(recommendation['total_cost']))
            m2.metric("Annual Savings", format_currency(recommendation['annual_savings_eur']))
            m3.metric("Payback Period", f"{recommendation['payback_period_years']:.1f} Yrs")
            m4.metric("Self-Sufficiency", f"{recommendation['self_sufficiency_pct']:.1f}%")

            st.markdown("---")
            v1, v2 = st.columns([1, 1])

            with v1:
                st.markdown("**Financial Summary**")
                df_financial = pd.DataFrame({'Metric': ["Total Project Cost", "Annual Maintenance", "Annual Financing Cost", "Annual Savings", "Payback Period (Years)"],
                                             'Value': [format_currency(recommendation['total_cost']), format_currency(recommendation['annual_maintenance_cost_eur']),
                                                       format_currency(recommendation['annual_financing_cost_eur']), format_currency(recommendation['annual_savings_eur']),
                                                       f"{recommendation['payback_period_years']:.1f}"]})
                st.table(df_financial)

                st.markdown("**Project Cost Breakdown**")
                costs = {'Solar Panels': recommendation['num_solar_panels'] * recommendation['model_constants']['COST_PER_SOLAR_PANEL'],
                         'Wind Turbines': recommendation['num_wind_turbines'] * recommendation['model_constants']['COST_PER_WIND_TURBINE'],
                         'Battery': recommendation['battery_kwh'] * recommendation['model_constants']['COST_PER_BATTERY_KWH']}
                hardware_cost = sum(costs.values())
                costs['Installation & Overheads'] = recommendation['total_cost'] - hardware_cost
                df_costs = pd.DataFrame(list(costs.items()), columns=['Category', 'Cost (€)'])
                fig_pie = px.pie(df_costs, values='Cost (€)', names='Category', title="Project Cost Distribution", hole=.3, color_discrete_sequence=px.colors.sequential.Tealgrn)
                st.plotly_chart(fig_pie, use_container_width=True)

            with v2:
                st.markdown("**ESG Score Radar Chart**")
                scores = recommendation.get('normalized_scores', {})
                if scores:
                    df_scores = pd.DataFrame(dict(r=list(scores.values()), theta=list(scores.keys())))
                    fig_radar = go.Figure(data=go.Scatterpolar(r=df_scores['r'], theta=df_scores['theta'], fill='toself'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
                    st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown("**Detailed ESG KPI Breakdown**")
                kpis = recommendation.get('raw_kpis', {})
                e_kpis, s_kpis, g_kpis = st.columns(3)
                with e_kpis:
                    with st.expander("🌳 Environmental KPIs", expanded=True):
                        st.table(pd.DataFrame({'KPI': ['CO2 Reduction (tons/yr)', 'Land Use (sqm)', 'Water Savings (m3/yr)', 'Energy Waste (%)'],
                                               'Value': [format_number(kpis.get('env_co2_reduction_tons_yr', 0)), format_number(kpis.get('env_land_use_sqm', 0)),
                                                         format_number(kpis.get('env_water_savings_m3_yr', 0)), f"{kpis.get('env_waste_factor_pct', 0):.1f}"]}))
                with s_kpis:
                    with st.expander("🤝 Social KPIs", expanded=True):
                        st.table(pd.DataFrame({'KPI': ['Local Jobs Created', 'Energy Resilience (hrs)', 'Grid Strain Reduction (%)'],
                                               'Value': [f"{kpis.get('soc_local_jobs_fte', 0):.2f}", f"{kpis.get('soc_energy_resilience_hrs', 0):.1f}",
                                                         f"{kpis.get('soc_grid_strain_reduction_pct', 0):.1f}"]}))
                with g_kpis:
                    with st.expander("🏛️ Governance KPIs", expanded=True):
                        st.table(pd.DataFrame({'KPI': ['Payback Plausibility (1-10)', 'Supply Chain Score (1-10)'],
                                               'Value': [f"{kpis.get('gov_payback_plausibility_score', 0):.1f}", f"{kpis.get('gov_supply_chain_transparency_score', 0):.1f}"]}))

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

with tab_predictor:
    st.header("⚡ ML Fast Predictor")
    st.success("✅ **System Ready:** The ML Models have been trained.")
    # Add ML prediction functionality here if desired

with tab_advisor:
    st.header("🤖 AI Advisor")
    user_query = st.text_input("Ask me anything about your HRES project...")
    if user_query:
        try:
            response = requests.post(f"{API_BASE_URL}/chat", json={"query": user_query})
            response.raise_for_status()
            ai_response = response.json().get("response", "I'm sorry, I couldn't generate a response.")
            st.markdown(ai_response)
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the AI Advisor: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

with tab_about:
    st.header("ℹ️ About")
    st.markdown("This application is a prototype...")