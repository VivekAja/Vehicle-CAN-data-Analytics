"""
Vehicle Predictive Maintenance — Inference Pipeline
----------------------------------------------------
Loads trained XGBoost model and Isolation Forest models,
runs feature engineering on raw sensor input,
outputs a structured maintenance alert.

Usage:
    python src/predict.py                    # runs demo on sample rows
    python src/predict.py --vehicle VEH0023  # filter to a specific vehicle
"""

import pandas as pd
import numpy as np
import joblib
import shap
import argparse
import os

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'df_anomaly.csv')

XGB_PATH         = os.path.join(MODEL_DIR, 'xgb_any_failure.pkl')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.pkl')
IF_MODELS_PATH   = os.path.join(MODEL_DIR, 'isolation_forests.pkl')

# ── Feature engineering (mirrors NB02 exactly) ────────────────────────────────
def engineer_features(row: pd.Series) -> pd.Series:
    """
    Apply domain-specific feature engineering to a single sensor reading.
    Rolling features default to the raw value (mean) and 0 (std)
    since we have no historical window for single-row inference.
    In production, pass a 5-row window and compute properly.
    """
    r = row.copy()

    # --- Engine ---
    r['thermal_stress']              = r['engine_temp_c'] * r['engine_load_percent'] / 100
    r['temp_delta_engine_coolant']   = r['engine_temp_c'] - r['coolant_temp_c']
    r['oil_pressure_per_rpm']        = r['oil_pressure_psi'] / (r['engine_rpm'] + 1)

    # --- Brake ---
    wheel_cols = ['wheel_speed_fl_kph', 'wheel_speed_fr_kph',
                  'wheel_speed_rl_kph', 'wheel_speed_rr_kph']
    wheel_vals = r[wheel_cols]
    r['wheel_speed_mean']   = wheel_vals.mean()
    r['wheel_speed_std']    = wheel_vals.std()
    r['brake_stress_event'] = r['vehicle_speed_kph'] * r['brake_pedal_pos_percent'] / 100

    # --- Battery ---
    r['alternator_deficit'] = r['battery_voltage_v'] - r['alternator_output_v']
    r['charge_efficiency']  = r['battery_charge_percent'] * r['battery_health_percent'] / 100

    # --- Rolling features — single row fallback (mean = value, std = 0) ---
    rolling_signals = [
        'engine_temp_c', 'engine_rpm', 'oil_pressure_psi', 'coolant_temp_c',
        'exhaust_gas_temp_c', 'brake_temp_c', 'brake_pad_wear_mm',
        'brake_fluid_level_psi', 'battery_voltage_v', 'battery_charge_percent',
        'battery_health_percent', 'alternator_output_v'
    ]
    for col in rolling_signals:
        r[f'{col}_roll_mean'] = r[col]   # single point → mean = itself
        r[f'{col}_roll_std']  = 0.0      # single point → no deviation

    return r


# ── Subsystem status via Isolation Forest ─────────────────────────────────────
def get_subsystem_status(row_engineered: pd.Series, if_models: dict) -> dict:
    """Run IF anomaly scoring per subsystem. Returns score + flag per system."""
    status = {}

    feature_sets = {
        'Engine': [
            'engine_temp_c', 'engine_rpm', 'oil_pressure_psi', 'coolant_temp_c',
            'exhaust_gas_temp_c', 'engine_load_percent', 'fuel_consumption_lph',
            'engine_temp_c_roll_mean', 'engine_temp_c_roll_std',
            'engine_rpm_roll_mean', 'engine_rpm_roll_std',
            'oil_pressure_psi_roll_mean', 'oil_pressure_psi_roll_std',
            'thermal_stress', 'temp_delta_engine_coolant', 'oil_pressure_per_rpm'
        ],
        'Brake': [
            'brake_fluid_level_psi', 'brake_pad_wear_mm', 'brake_temp_c',
            'abs_fault_indicator', 'brake_pedal_pos_percent',
            'brake_temp_c_roll_mean', 'brake_temp_c_roll_std',
            'brake_pad_wear_mm_roll_mean', 'brake_pad_wear_mm_roll_std',
            'wheel_speed_std', 'wheel_speed_mean', 'brake_stress_event'
        ],
        'Battery': [
            'battery_voltage_v', 'battery_current_a', 'battery_temp_c',
            'alternator_output_v', 'battery_charge_percent', 'battery_health_percent',
            'battery_voltage_v_roll_mean', 'battery_voltage_v_roll_std',
            'battery_charge_percent_roll_mean', 'battery_charge_percent_roll_std',
            'battery_health_percent_roll_mean', 'battery_health_percent_roll_std',
            'alternator_deficit', 'charge_efficiency'
        ]
    }

    for system, feats in feature_sets.items():
        scaler, iso = if_models[system]['scaler'], if_models[system]['model']
        X = scaler.transform(row_engineered[feats].values.reshape(1, -1))
        score = iso.decision_function(X)[0]
        flag  = iso.predict(X)[0]  # -1 = anomaly, 1 = normal
        status[system] = {
            'score':    round(score, 3),
            'anomalous': flag == -1
        }

    return status


# ── SHAP explanation for top driving features ─────────────────────────────────
def get_shap_explanation(row_features: pd.DataFrame,
                          xgb_model,
                          feature_cols: list,
                          top_n: int = 3) -> list:
    """Returns top N features with their SHAP values for this prediction."""
    explainer   = shap.TreeExplainer(xgb_model)
    shap_vals   = explainer.shap_values(row_features[feature_cols])[0]
    shap_series = pd.Series(shap_vals, index=feature_cols)
    top_features = shap_series.abs().nlargest(top_n).index.tolist()

    return [
        {
            'feature': f,
            'value':   round(row_features[f].iloc[0], 3),
            'shap':    round(shap_series[f], 4)
        }
        for f in top_features
    ]


# ── Risk level thresholds ─────────────────────────────────────────────────────
def risk_level(prob: float) -> tuple:
    if prob >= 0.80:
        return '⚠️  HIGH RISK',   'Immediate inspection required'
    elif prob >= 0.40:
        return '🔶 MEDIUM RISK', 'Schedule maintenance within 48h'
    else:
        return '✅ LOW RISK',    'Continue normal monitoring'


# ── Main prediction function ──────────────────────────────────────────────────
def predict(raw_row: pd.Series,
            xgb_model,
            if_models: dict,
            feature_cols: list,
            verbose: bool = True) -> dict:
    """
    Full inference pipeline for one vehicle sensor reading.

    Parameters
    ----------
    raw_row     : pd.Series — raw sensor values (one row from telemetry)
    xgb_model   : trained XGBClassifier
    if_models   : dict of {subsystem: {scaler, model}} Isolation Forests
    feature_cols: list of feature column names used during training
    verbose     : print formatted alert card

    Returns
    -------
    dict with all prediction outputs
    """
    # Step 1 — feature engineering
    engineered = engineer_features(raw_row)
    X_row = pd.DataFrame([engineered])

    # Step 2 — overall failure probability
    prob = xgb_model.predict_proba(X_row[feature_cols])[0][1]

    # Step 3 — subsystem anomaly status
    subsystem_status = get_subsystem_status(engineered, if_models)

    # Step 4 — SHAP explanation
    top_signals = get_shap_explanation(X_row, xgb_model, feature_cols)

    # Step 5 — risk level
    level, recommendation = risk_level(prob)

    result = {
        'vehicle_id':        raw_row.get('vehicle_id', 'N/A'),
        'brand':             raw_row.get('brand', 'N/A'),
        'timestamp':         raw_row.get('timestamp', 'N/A'),
        'failure_prob':      round(prob, 4),
        'risk_level':        level,
        'recommendation':    recommendation,
        'subsystem_status':  subsystem_status,
        'top_signals':       top_signals
    }

    if verbose:
        _print_alert(result)

    return result


# ── Formatted output ──────────────────────────────────────────────────────────
def _print_alert(result: dict):
    width = 62
    print('╔' + '═' * width + '╗')
    print('║' + '   VEHICLE PREDICTIVE MAINTENANCE ALERT'.center(width) + '║')
    print('╚' + '═' * width + '╝')
    print(f"  Vehicle  : {result['vehicle_id']}  │  "
          f"Brand: {result['brand']}  │  {result['timestamp']}")
    print()
    print(f"  OVERALL RISK SCORE : {result['failure_prob']:.2f}  →  {result['risk_level']}")
    print()
    print('  SUBSYSTEM STATUS')
    for system, info in result['subsystem_status'].items():
        icon  = '🔴' if info['anomalous'] else '🟢'
        state = 'ANOMALOUS' if info['anomalous'] else 'Normal   '
        print(f"    {icon} {system:8s}: {state}  (IF score: {info['score']:+.2f})")
    print()
    print('  TOP SIGNALS DRIVING THIS PREDICTION')
    for sig in result['top_signals']:
        direction = '↑' if sig['shap'] > 0 else '↓'
        print(f"    {direction} {sig['feature']:35s}"
              f"= {sig['value']:<10}  [{sig['shap']:+.3f} toward failure]")
    print()
    print(f"  RECOMMENDATION : {result['recommendation']}")
    print('─' * (width + 2))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Vehicle failure risk predictor')
    parser.add_argument('--vehicle', type=str, default=None,
                        help='Filter to a specific vehicle_id')
    parser.add_argument('--n',       type=int, default=3,
                        help='Number of sample rows to predict (default: 3)')
    args = parser.parse_args()

    # Load artifacts
    print('Loading models...')
    xgb_model    = joblib.load(XGB_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    if_models    = joblib.load(IF_MODELS_PATH)
    print(f'  XGBoost model    : {XGB_PATH}')
    print(f'  Feature columns  : {len(feature_cols)} features')
    print(f'  Isolation Forests: {list(if_models.keys())}')
    print()

    # Load data — pick a mix of failure and normal rows for demo
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

    if args.vehicle:
        df = df[df['vehicle_id'] == args.vehicle]
        if df.empty:
            print(f'No records found for vehicle {args.vehicle}')
            return

    # Show failure rows first, then pad with normal rows
    failure_rows = df[df['any_failure'] == 1].head(args.n)
    normal_rows  = df[df['any_failure'] == 0].head(max(0, args.n - len(failure_rows)))
    sample       = pd.concat([failure_rows, normal_rows]).reset_index(drop=True)

    print(f'Running inference on {len(sample)} sample rows '
          f'({len(failure_rows)} failures, {len(normal_rows)} normal)...')
    print()

    for _, row in sample.iterrows():
        predict(row, xgb_model, if_models, feature_cols, verbose=True)


if __name__ == '__main__':
    main()