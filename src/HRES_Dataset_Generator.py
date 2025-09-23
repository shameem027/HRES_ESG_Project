# --- File: src/HRES_Dataset_Generator.py (Definitive Final Version) ---
import pandas as pd; import numpy as np; from tqdm import tqdm; import os
# --- Constants ---
COST_PER_SOLAR_PANEL, COST_PER_WIND_TURBINE, COST_PER_BATTERY_KWH = 100, 200, 200
INSTALLATION_COST_FACTOR, ENGINEERING_CONSULTING_COST_FACTOR, PERMITTING_LEGAL_COST_FACTOR, OTHER_COMPONENTS_COST_FACTOR = 0.60, 0.15, 0.05, 0.10
ANNUAL_OM_RATE, PROJECT_LIFETIME_YEARS, BATTERY_LIFETIME_YEARS, FINANCING_INTEREST_RATE = 0.02, 25, 8, 0.07
HOURS_IN_YEAR, BATTERY_ROUNDTRIP_EFFICIENCY, BATTERY_MIN_SOC_PERCENT, SOLAR_PANEL_DEGRADATION_PER_YEAR = 8760, 0.90, 0.20, 0.005
PRICE_PONTA, PRICE_CHEIAS, PRICE_VAZIO, GRID_EXPORT_PRICE = 0.28, 0.18, 0.11, 0.05
CO2_REDUCTION_PER_KWH, WATER_SAVED_PER_KWH_RE, LAND_USE_SOLAR_SQM_PER_PANEL, LAND_USE_WIND_SQM_PER_TURBINE, JOBS_PER_1M_COST = 0.000199, 0.002, 1.7, 5, 1.2
def get_tou_price_schedule_portugal():
    price_schedule = np.full(HOURS_IN_YEAR, PRICE_VAZIO);
    for day in range(365):
        if day % 7 < 5: start_hour = day * 24; price_schedule[start_hour + 7:start_hour + 9] = PRICE_CHEIAS; price_schedule[start_hour + 9:start_hour + 12] = PRICE_PONTA; price_schedule[start_hour + 18:start_hour + 21] = PRICE_PONTA; price_schedule[start_hour + 21:start_hour + 23] = PRICE_CHEIAS
    return price_schedule
def generate_load_profile(annual_demand_kwh, profile_type):
    profile_map = {'office': [0.3]*8 + [0.8]*2 + [1.0]*8 + [0.7]*2 + [0.3]*4, 'hospital': [0.7]*7 + [0.9]*12 + [0.8]*5, 'university': [0.4]*7 + [0.7]*2 + [0.9]*8 + [0.8]*3 + [0.5]*4, 'data_center': [0.95]*24, 'industrial': [0.4]*7 + [1.0]*10 + [0.4]*7}
    weekday_profile = np.array(profile_map.get(profile_type)); weekend_profile = weekday_profile * 0.5 if profile_type not in ['hospital', 'data_center'] else weekday_profile
    yearly_profile = np.tile(np.concatenate((np.tile(weekday_profile, 5), np.tile(weekend_profile, 2))), 52); yearly_profile = np.append(yearly_profile, weekday_profile)[:HOURS_IN_YEAR]; yearly_profile *= (1 + np.random.normal(0, 0.03, len(yearly_profile)))
    return (yearly_profile / yearly_profile.sum()) * annual_demand_kwh
def generate_solar_profile(num_panels):
    kw_capacity = num_panels * 0.5; daily_solar_shape = np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, 24) - np.pi / 1.5)); yearly_profile = np.tile(daily_solar_shape, 365)[:HOURS_IN_YEAR]; seasonal_weather = 1 - 0.3 * np.cos(np.linspace(0, 2 * np.pi, HOURS_IN_YEAR)); random_clouds = 1 - np.random.gamma(0.5, 0.2, HOURS_IN_YEAR)
    return yearly_profile * seasonal_weather * np.clip(random_clouds, 0.1, 1.0) * kw_capacity
def generate_wind_profile(num_turbines):
    kw_capacity = num_turbines * 1.0; random_profile = np.random.rand(HOURS_IN_YEAR); smoothed_profile = pd.Series(random_profile).rolling(window=72, center=True, min_periods=1).mean().bfill().ffill().values; seasonal_multiplier = 1.3 - 0.5 * np.sin(np.linspace(0, 2 * np.pi, HOURS_IN_YEAR) + np.pi / 2)
    return smoothed_profile * seasonal_multiplier * kw_capacity * 0.25 * 2.5
def simulate_hres_yearly(config, load_profile, price_schedule, total_gen):
    battery_soc, min_soc = config['battery_kwh'] * 0.5, config['battery_kwh'] * BATTERY_MIN_SOC_PERCENT
    total_grid_import, total_grid_export, total_curtailment, cost_of_grid_energy, revenue_from_export = 0.0, 0.0, 0.0, 0.0, 0.0
    for hour in range(HOURS_IN_YEAR):
        net_power = total_gen[hour] - load_profile[hour]
        if net_power > 0:
            charge_amount = min(net_power, config['battery_kwh'] - battery_soc); battery_soc += charge_amount * np.sqrt(BATTERY_ROUNDTRIP_EFFICIENCY); surplus = net_power - charge_amount; export_limit = (config['num_solar_panels'] * 0.5 + config['num_wind_turbines'] * 1.0) * 0.5
            grid_export = min(surplus, export_limit); total_grid_export += grid_export; total_curtailment += (surplus - grid_export); revenue_from_export += grid_export * GRID_EXPORT_PRICE
        else:
            discharge_amount = min(abs(net_power), battery_soc - min_soc); battery_soc -= discharge_amount; net_power += discharge_amount / np.sqrt(BATTERY_ROUNDTRIP_EFFICIENCY)
            if net_power < 0: grid_import = abs(net_power); total_grid_import += grid_import; cost_of_grid_energy += grid_import * price_schedule[hour]
    total_load = load_profile.sum(); cost_without_hres = (load_profile * price_schedule).sum(); total_initial_capex = config['total_cost']
    annual_om_cost = total_initial_capex * ANNUAL_OM_RATE; num_battery_replacements = int(np.floor(PROJECT_LIFETIME_YEARS / BATTERY_LIFETIME_YEARS)); total_battery_replacement_cost = num_battery_replacements * config['battery_kwh'] * COST_PER_BATTERY_KWH
    annual_financing_cost = total_initial_capex * FINANCING_INTEREST_RATE; annual_amortized_battery_replacement_cost = total_battery_replacement_cost / PROJECT_LIFETIME_YEARS
    annual_operating_cost_total = cost_of_grid_energy + annual_om_cost + annual_amortized_battery_replacement_cost + annual_financing_cost - revenue_from_export
    annual_savings = cost_without_hres - annual_operating_cost_total; payback_period = total_initial_capex / (annual_savings + 1e-9)
    payback_period = np.clip(payback_period, 0, PROJECT_LIFETIME_YEARS * 2) if annual_savings > 0 else PROJECT_LIFETIME_YEARS * 5
    return {'annual_kwh_generated': round(total_gen.sum()), 'self_sufficiency_pct': round(((total_load - total_grid_import) / (total_load + 1e-6)) * 100, 1), 'annual_savings_eur': round(annual_savings), 'payback_period_years': round(payback_period, 1), 'annual_kwh_exported': round(total_grid_export), 'annual_kwh_curtailed': round(total_curtailment), 'annual_maintenance_cost_eur': round(annual_om_cost), 'annual_financing_cost_eur': round(annual_financing_cost)}
def main():
    print("🚀 Starting HRES Dataset Generation..."); np.random.seed(42)
    scenarios = {"Small_Office": {"demand_kwh": 250000, "profile_type": "office"}, "Hospital": {"demand_kwh": 1500000, "profile_type": "hospital"}, "University_Campus": {"demand_kwh": 3000000, "profile_type": "university"}, "Industrial_Facility": {"demand_kwh": 5000000, "profile_type": "industrial"}, "Data_Center": {"demand_kwh": 10000000, "profile_type": "data_center"}}
    solar_options, wind_options, battery_options = np.linspace(100, 8000, 4, dtype=int), np.linspace(5, 200, 4, dtype=int), np.linspace(250, 10000, 4, dtype=int)
    all_results, price_schedule = [], get_tou_price_schedule_portugal()
    base_configs = [{'num_solar_panels': s, 'num_wind_turbines': w, 'battery_kwh': b} for s in solar_options for w in wind_options for b in battery_options if s > 0 or w > 0]
    for scenario_name, params in scenarios.items():
        load_profile = generate_load_profile(params['demand_kwh'], params['profile_type'])
        for config_base in tqdm(base_configs, desc=f"Simulating for {scenario_name}"):
            config = config_base.copy(); config['scenario_name'], config['annual_demand_kwh'] = scenario_name, params['demand_kwh']
            hardware_cost = (config['num_solar_panels'] * COST_PER_SOLAR_PANEL + config['num_wind_turbines'] * COST_PER_WIND_TURBINE + config['battery_kwh'] * COST_PER_BATTERY_KWH)
            installation_cost = hardware_cost * INSTALLATION_COST_FACTOR; consulting_fee = hardware_cost * ENGINEERING_CONSULTING_COST_FACTOR; permit_fee = hardware_cost * PERMITTING_LEGAL_COST_FACTOR; other_costs = hardware_cost * OTHER_COMPONENTS_COST_FACTOR
            config['total_cost'] = hardware_cost + installation_cost + consulting_fee + permit_fee + other_costs
            solar_gen, wind_gen = generate_solar_profile(config['num_solar_panels']), generate_wind_profile(config['num_wind_turbines']); total_gen = (solar_gen + wind_gen) * (1 - SOLAR_PANEL_DEGRADATION_PER_YEAR)
            config.update(simulate_hres_yearly(config, load_profile, price_schedule, total_gen))
            config['env_co2_reduction_tons_yr'] = round(config['annual_kwh_generated'] * CO2_REDUCTION_PER_KWH); config['env_land_use_sqm'] = round(config['num_solar_panels'] * LAND_USE_SOLAR_SQM_PER_PANEL + config['num_wind_turbines'] * LAND_USE_WIND_SQM_PER_TURBINE); config['env_water_savings_m3_yr'] = round(config['annual_kwh_generated'] * WATER_SAVED_PER_KWH_RE); config['env_waste_factor_pct'] = round((config['annual_kwh_curtailed'] / (config['annual_kwh_generated'] + 1e-6)) * 100, 1)
            config['soc_local_jobs_fte'] = round((config['total_cost'] / 1_000_000) * JOBS_PER_1M_COST, 2); daily_avg_demand_kwh = params['demand_kwh'] / HOURS_IN_YEAR; config['soc_energy_resilience_hrs'] = round((config['battery_kwh'] * (1 - BATTERY_MIN_SOC_PERCENT)) / (daily_avg_demand_kwh + 1e-6), 1); config['soc_grid_strain_reduction_pct'] = round((config['annual_kwh_exported'] / (config['annual_kwh_generated'] + 1e-6)) * 100, 1)
            config['gov_payback_plausibility_score'] = round(np.clip(PROJECT_LIFETIME_YEARS / (config['payback_period_years'] + 1e-6), 1, 10), 1); config['gov_supply_chain_transparency_score'] = round(np.clip((config['num_solar_panels'] * 8 + config['num_wind_turbines'] * 6) / (config['num_solar_panels'] + config['num_wind_turbines'] + 1e-6), 1, 10), 1)
            all_results.append(config)
    df = pd.DataFrame(all_results)
    output_path = os.path.join(os.path.dirname(__file__), 'HRES_Dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully generated {len(df)} realistic simulation runs. Dataset saved to {output_path}")
if __name__ == "__main__": main()