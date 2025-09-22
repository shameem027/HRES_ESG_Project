# --- File: src/HRES_Dataset_Generator.py ---
# --- Author: Md Shameem Hossain ---
# --- Purpose: Generates a comprehensive dataset of HRES solutions with detailed financial and ESG metrics. ---

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# -----------------------------------------------------------------------------
# 1. DEFINE SIMULATION CONSTANTS (ADJUSTED FOR REALISM & USER REQUIREMENTS)
# -----------------------------------------------------------------------------
# Component Costs (User-specified base values)
COST_PER_SOLAR_PANEL = 100  # Assuming 0.5kW/panel nominal capacity
COST_PER_WIND_TURBINE = 200  # Assuming 1.0kW/turbine nominal capacity
COST_PER_BATTERY_KWH = 200

# Additional Project Costs & Factors (as % of Initial Hardware Cost)
INSTALLATION_OVERHEAD_FACTOR = 0.60
ENGINEERING_CONSULTING_COST_FACTOR = 0.15
PERMITTING_LEGAL_COST_FACTOR = 0.05
OTHER_COMPONENTS_COST_FACTOR = 0.10

# Annual Operating Costs
ANNUAL_OM_RATE = 0.02  # 2% of Total Initial CAPEX

# Project Lifecycles
PROJECT_LIFETIME_YEARS = 25
BATTERY_LIFETIME_YEARS = 8  # Assumes battery replacement needed twice

# Financing
FINANCING_INTEREST_RATE = 0.07

# Performance & Efficiency
HOURS_IN_YEAR = 8760
BATTERY_ROUNDTRIP_EFFICIENCY = 0.90
BATTERY_MIN_SOC_PERCENT = 0.20
SOLAR_PANEL_DEGRADATION_PER_YEAR = 0.005

# Portuguese Time-of-Use (TOU) Pricing (€/kWh)
PRICE_PONTA = 0.28  # Peak
PRICE_CHEIAS = 0.18  # Mid-Peak
PRICE_VAZIO = 0.11  # Off-Peak
GRID_EXPORT_PRICE = 0.05

# ESG KPI Conversion Factors
CO2_REDUCTION_PER_KWH = 0.000199
WATER_SAVED_PER_KWH_RE = 0.002
LAND_USE_SOLAR_SQM_PER_PANEL = 1.7
LAND_USE_WIND_SQM_PER_TURBINE = 5
JOBS_PER_1M_COST = 1.2


# -----------------------------------------------------------------------------
# 2. GENERATE DYNAMIC HOURLY PROFILES
# -----------------------------------------------------------------------------
def get_tou_price_schedule_portugal():
    price_schedule = np.full(HOURS_IN_YEAR, PRICE_VAZIO)
    for day in range(365):
        if day % 7 < 5:  # Weekdays (Monday=0 to Thursday=4)
            start_hour = day * 24
            price_schedule[start_hour + 7:start_hour + 9] = PRICE_CHEIAS
            price_schedule[start_hour + 9:start_hour + 12] = PRICE_PONTA
            price_schedule[start_hour + 18:start_hour + 21] = PRICE_PONTA
            price_schedule[start_hour + 21:start_hour + 23] = PRICE_CHEIAS
    return price_schedule


def generate_load_profile(annual_demand_kwh, profile_type='office'):
    profile_map = {
        'office': np.array([0.3] * 8 + [0.8] * 2 + [1.0] * 8 + [0.7] * 2 + [0.3] * 4),
        'hospital': np.array([0.7] * 7 + [0.9] * 12 + [0.8] * 5),
        'university': np.array([0.4] * 7 + [0.7] * 2 + [0.9] * 8 + [0.8] * 3 + [0.5] * 4),
        'data_center': np.array([0.95] * 24),
        'industrial': np.array([0.4] * 7 + [1.0] * 10 + [0.4] * 7)
    }
    weekday_profile = profile_map.get(profile_type)
    weekend_profile = weekday_profile * 0.5 if profile_type not in ['hospital', 'data_center'] else weekday_profile

    yearly_profile = np.tile(np.concatenate((np.tile(weekday_profile, 5), np.tile(weekend_profile, 2))), 52)
    yearly_profile = np.append(yearly_profile, weekday_profile)[:HOURS_IN_YEAR]
    yearly_profile *= (1 + np.random.normal(0, 0.03, len(yearly_profile)))
    return (yearly_profile / yearly_profile.sum()) * annual_demand_kwh


def generate_solar_profile(num_panels):
    kw_capacity = num_panels * 0.5
    daily_solar_shape = np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, 24) - np.pi / 1.5))
    yearly_profile = np.tile(daily_solar_shape, 365)[:HOURS_IN_YEAR]
    seasonal_weather = 1 - 0.3 * np.cos(np.linspace(0, 2 * np.pi, HOURS_IN_YEAR))
    random_clouds = 1 - np.random.gamma(0.5, 0.2, HOURS_IN_YEAR)
    return yearly_profile * seasonal_weather * np.clip(random_clouds, 0.1, 1.0) * kw_capacity


def generate_wind_profile(num_turbines):
    kw_capacity = num_turbines * 1.0
    random_profile = np.random.rand(HOURS_IN_YEAR)
    smoothed_profile = pd.Series(random_profile).rolling(window=72, center=True,
                                                         min_periods=1).mean().bfill().ffill().values
    seasonal_multiplier = 1.3 - 0.5 * np.sin(np.linspace(0, 2 * np.pi, HOURS_IN_YEAR) + np.pi / 2)
    return smoothed_profile * seasonal_multiplier * kw_capacity * 0.25 * 2.5


# -----------------------------------------------------------------------------
# 3. THE HOURLY SIMULATION ENGINE
# -----------------------------------------------------------------------------
def simulate_hres_yearly(config, load_profile, price_schedule, total_gen, solar_gen_profile, wind_gen_profile):
    battery_soc = config['battery_kwh'] * 0.5
    min_soc = config['battery_kwh'] * BATTERY_MIN_SOC_PERCENT
    total_grid_import, total_grid_export, total_curtailment = 0.0, 0.0, 0.0
    cost_of_grid_energy, revenue_from_export = 0.0, 0.0

    for hour in range(HOURS_IN_YEAR):
        net_power = total_gen[hour] - load_profile[hour]
        if net_power > 0:  # Surplus generation
            charge_amount = min(net_power, config['battery_kwh'] - battery_soc)
            battery_soc += charge_amount * np.sqrt(BATTERY_ROUNDTRIP_EFFICIENCY)
            surplus = net_power - charge_amount
            export_limit = (config['num_solar_panels'] * 0.5 + config['num_wind_turbines'] * 1.0) * 0.5
            grid_export = min(surplus, export_limit)
            total_grid_export += grid_export
            total_curtailment += (surplus - grid_export)
            revenue_from_export += grid_export * GRID_EXPORT_PRICE
        else:  # Power deficit
            discharge_amount = min(abs(net_power), battery_soc - min_soc)
            battery_soc -= discharge_amount
            net_power += discharge_amount / np.sqrt(BATTERY_ROUNDTRIP_EFFICIENCY)
            if net_power < 0:
                grid_import = abs(net_power)
                total_grid_import += grid_import
                cost_of_grid_energy += grid_import * price_schedule[hour]

    total_load = load_profile.sum()
    cost_without_hres = (load_profile * price_schedule).sum()

    # Calculate lifetime and annual costs
    initial_hardware_cost_raw = (config['num_solar_panels'] * COST_PER_SOLAR_PANEL +
                                 config['num_wind_turbines'] * COST_PER_WIND_TURBINE +
                                 config['battery_kwh'] * COST_PER_BATTERY_KWH)
    total_initial_capex = config['total_cost']  # Already calculated in main loop
    annual_om_cost = total_initial_capex * ANNUAL_OM_RATE
    num_battery_replacements = max(0, int(np.floor(PROJECT_LIFETIME_YEARS / BATTERY_LIFETIME_YEARS)) - 1)
    total_battery_replacement_cost = num_battery_replacements * (config['battery_kwh'] * COST_PER_BATTERY_KWH)
    annual_amortized_battery_replacement_cost = total_battery_replacement_cost / PROJECT_LIFETIME_YEARS
    annual_financing_cost = total_initial_capex * FINANCING_INTEREST_RATE

    annual_operating_cost_total = cost_of_grid_energy + annual_om_cost + annual_amortized_battery_replacement_cost + annual_financing_cost - revenue_from_export
    annual_savings = cost_without_hres - annual_operating_cost_total

    payback_period = total_initial_capex / (annual_savings + 1e-6)  # Add epsilon to avoid division by zero
    payback_period = np.clip(payback_period, 0, PROJECT_LIFETIME_YEARS * 2)

    return {
        'annual_kwh_generated': round(total_gen.sum()),
        'self_sufficiency_pct': round(((total_load - total_grid_import) / (total_load + 1e-6)) * 100, 1),
        'annual_savings_eur': round(annual_savings),
        'payback_period_years': round(payback_period, 1),
        'annual_kwh_exported': round(total_grid_export),
        'annual_kwh_curtailed': round(total_curtailment),
        'wind_generation_kwh': round(wind_gen_profile.sum()),
        'solar_generation_kwh': round(solar_gen_profile.sum()),
        'annual_maintenance_cost_eur': round(annual_om_cost),
        'annual_amortized_battery_replacement_cost_eur': round(annual_amortized_battery_replacement_cost),
        'annual_financing_cost_eur': round(annual_financing_cost)
    }


# -----------------------------------------------------------------------------
# 4. MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    print("🚀 Starting HRES Dataset Generation...")
    # Set a random seed for reproducibility
    np.random.seed(42)

    scenarios = {
        "Small_Office": {"demand_kwh": 250000, "profile_type": "office"},
        "Hospital": {"demand_kwh": 1500000, "profile_type": "hospital"},
        "University_Campus": {"demand_kwh": 3000000, "profile_type": "university"},
        "Industrial_Facility": {"demand_kwh": 5000000, "profile_type": "industrial"},
        "Data_Center": {"demand_kwh": 10000000, "profile_type": "data_center"}
    }

    solar_options = np.linspace(100, 8000, 10, dtype=int)
    wind_options = np.linspace(5, 200, 10, dtype=int)
    battery_options = np.linspace(250, 10000, 10, dtype=int)

    all_results = []
    price_schedule = get_tou_price_schedule_portugal()

    base_configs = [{'num_solar_panels': s, 'num_wind_turbines': w, 'battery_kwh': b}
                    for s in solar_options for w in wind_options for b in battery_options if s > 0 or w > 0]

    print(f"Generating a dataset with {len(base_configs)} base configurations per scenario.")

    for scenario_name, params in scenarios.items():
        load_profile = generate_load_profile(params['demand_kwh'], params['profile_type'])
        for config_base in tqdm(base_configs, desc=f"Simulating for {scenario_name}"):
            config = config_base.copy()
            config['scenario_name'] = scenario_name
            config['annual_demand_kwh'] = params['demand_kwh']

            initial_hardware_cost = (config['num_solar_panels'] * COST_PER_SOLAR_PANEL +
                                     config['num_wind_turbines'] * COST_PER_WIND_TURBINE +
                                     config['battery_kwh'] * COST_PER_BATTERY_KWH)

            config['total_cost'] = initial_hardware_cost * (1 + INSTALLATION_OVERHEAD_FACTOR +
                                                            ENGINEERING_CONSULTING_COST_FACTOR +
                                                            PERMITTING_LEGAL_COST_FACTOR +
                                                            OTHER_COMPONENTS_COST_FACTOR)

            solar_gen = generate_solar_profile(config['num_solar_panels']) * (1 - SOLAR_PANEL_DEGRADATION_PER_YEAR)
            wind_gen = generate_wind_profile(config['num_wind_turbines'])
            total_gen = solar_gen + wind_gen

            sim_results = simulate_hres_yearly(config, load_profile, price_schedule, total_gen, solar_gen, wind_gen)
            config.update(sim_results)

            # ESG KPIs
            config['env_co2_reduction_tons_yr'] = round(config['annual_kwh_generated'] * CO2_REDUCTION_PER_KWH)
            config['env_land_use_sqm'] = round(config['num_solar_panels'] * LAND_USE_SOLAR_SQM_PER_PANEL + config[
                'num_wind_turbines'] * LAND_USE_WIND_SQM_PER_TURBINE)
            config['env_water_savings_m3_yr'] = round(config['annual_kwh_generated'] * WATER_SAVED_PER_KWH_RE)
            config['env_waste_factor_pct'] = round(
                (config['annual_kwh_curtailed'] / (config['annual_kwh_generated'] + 1e-6)) * 100, 1)
            config['env_degradation_rate'] = SOLAR_PANEL_DEGRADATION_PER_YEAR
            config['soc_local_jobs_fte'] = round((config['total_cost'] / 1_000_000) * JOBS_PER_1M_COST, 2)
            daily_avg_demand_kwh = params['demand_kwh'] / HOURS_IN_YEAR
            config['soc_energy_resilience_hrs'] = round(
                (config['battery_kwh'] * (1 - BATTERY_MIN_SOC_PERCENT)) / (daily_avg_demand_kwh + 1e-6), 1)
            config['soc_grid_strain_reduction_pct'] = round(
                (config['annual_kwh_exported'] / (config['annual_kwh_generated'] + 1e-6)) * 100, 1)
            config['soc_community_investment_eur'] = round(config['total_cost'] * 0.001)
            config['soc_noise_level_impact_score'] = round(
                np.clip(10 - np.log1p(config['num_wind_turbines'] * 2), 1, 10), 1)
            config['gov_payback_plausibility_score'] = round(
                np.clip(PROJECT_LIFETIME_YEARS / (config['payback_period_years'] + 1e-6), 1, 10), 1)
            config['gov_supply_chain_transparency_score'] = round(np.clip(
                (config['num_solar_panels'] * 8 + config['num_wind_turbines'] * 6 + config['battery_kwh'] * 5) / (
                            config['num_solar_panels'] + config['num_wind_turbines'] * 2 + config[
                        'battery_kwh'] + 1e-6), 1, 10), 1)
            config['gov_regulatory_compliance_score'] = 10
            config['gov_stakeholder_reporting_score'] = 8
            config['gov_operational_risk_score'] = round(np.clip(10 - (
                        config['num_solar_panels'] / 2000 + config['num_wind_turbines'] / 50 + config[
                    'battery_kwh'] / 5000), 1, 10), 1)

            all_results.append(config)

    df = pd.DataFrame(all_results)
    # Save inside the 'src' directory, which is standard for shared data within the app
    output_path = os.path.join(os.path.dirname(__file__), 'HRES_Dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully generated {len(df)} simulation runs. Dataset saved to {output_path}")


if __name__ == "__main__":
    main()