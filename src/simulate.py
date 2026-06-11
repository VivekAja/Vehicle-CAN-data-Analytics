"""
Vehicle Predictive Maintenance — Sensor Simulation
---------------------------------------------------
Predict failure risk from raw sensor values directly.
No vehicle ID needed — pure signal-in, risk-out.

Presets:
    python src/simulate.py --preset normal
    python src/simulate.py --preset brake_wear
    python src/simulate.py --preset engine_stress
    python src/simulate.py --preset battery_drain
    python src/simulate.py --preset critical

Override any sensor on top of a preset:
    python src/simulate.py --preset normal --set brake_fluid_level_psi=200
    python src/simulate.py --preset normal --set engine_temp_c=130 oil_pressure_psi=15
    python src/simulate.py --preset brake_wear --set brake_pad_wear_mm=1.5
"""

import pandas as pd
import numpy as np
import joblib
import shap
import argparse
import os
import sys

BASE         = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE, '..', 'outputs', 'models')
XGB_PATH     = os.path.join(MODEL_DIR, 'xgb_any_failure.pkl')
FEAT_PATH    = os.path.join(MODEL_DIR, 'feature_cols.pkl')
IF_PATH      = os.path.join(MODEL_DIR, 'isolation_forests.pkl')

# ── Normal operating baseline (population medians from dataset) ───────────────
NORMAL_BASELINE = {
    # Engine
    'engine_temp_c':          88.0,
    'engine_rpm':            2100.0,
    'oil_pressure_psi':        55.0,
    'coolant_temp_c':          85.0,
    'fuel_level_percent':      65.0,
    'fuel_consumption_lph':     5.2,
    'engine_load_percent':     48.0,
    'throttle_pos_percent':    35.0,
    'air_flow_rate_gps':       18.0,
    'exhaust_gas_temp_c':     320.0,
    'vibration_level':          1.8,
    'engine_hours':           850.0,
    # Brake
    'brake_fluid_level_psi':  900.0,
    'brake_pad_wear_mm':        8.0,
    'brake_temp_c':           120.0,
    'abs_fault_indicator':      0,
    'brake_pedal_pos_percent': 15.0,
    # Wheels
    'wheel_speed_fl_kph':      80.0,
    'wheel_speed_fr_kph':      80.0,
    'wheel_speed_rl_kph':      80.0,
    'wheel_speed_rr_kph':      80.0,
    # Battery
    'battery_voltage_v':       12.8,
    'battery_current_a':       14.0,
    'battery_temp_c':          28.0,
    'alternator_output_v':     14.2,
    'battery_charge_percent':  78.0,
    'battery_health_percent':  88.0,
    # Vehicle / environment
    'vehicle_speed_kph':       80.0,
    'ambient_temp_c':          22.0,
    'humidity_percent':        55.0,
    'odometer_reading':     42000.0,
}

# ── Presets: deviations from baseline ────────────────────────────────────────
PRESETS = {
    'normal': {},   # pure baseline

    'brake_wear': {
        'brake_fluid_level_psi': 320.0,   # critically low
        'brake_pad_wear_mm':       1.8,   # heavily worn
        'brake_temp_c':          290.0,   # overheating
        'abs_fault_indicator':     1,     # ABS fault active
    },

    'engine_stress': {
        'engine_temp_c':         128.0,   # overheating
        'oil_pressure_psi':       16.0,   # dangerously low
        'vibration_level':         5.2,   # high vibration
        'exhaust_gas_temp_c':    680.0,   # very hot exhaust
        'engine_rpm':           4800.0,   # high RPM
        'coolant_temp_c':        118.0,   # coolant hot
    },

    'battery_drain': {
        'battery_voltage_v':      10.9,   # below threshold
        'battery_charge_percent': 12.0,   # almost dead
        'battery_health_percent': 42.0,   # degraded
        'alternator_output_v':    11.1,   # alternator failing
    },

    'critical': {
        # Everything going wrong simultaneously
        'brake_fluid_level_psi': 280.0,
        'brake_temp_c':          310.0,
        'engine_temp_c':         132.0,
        'oil_pressure_psi':       12.0,
        'battery_voltage_v':      10.8,
        'battery_health_percent': 35.0,
        'vibration_level':         6.1,
        'abs_fault_indicator':     1,
    },
}

# ── Normal ranges for context (approx 5th-95th percentile of healthy data) ───
NORMAL_RANGES = {
    'engine_temp_c':          (72,  105),
    'oil_pressure_psi':       (30,   80),
    'brake_fluid_level_psi':  (700, 1100),
    'brake_temp_c':           (70,  195),
    'brake_pad_wear_mm':      (3.5,  13),
    'battery_voltage_v':      (11.8, 14.5),
    'battery_charge_percent': (35,   98),
    'battery_health_percent': (60,  100),
    'vibration_level':        (0.5,  3.5),
}

# ── Feature engineering (identical to predict.py / NB02) ─────────────────────
def engineer(row: pd.Series) -> pd.Series:
    r = row.copy()
    r['thermal_stress']            = r['engine_temp_c'] * r['engine_load_percent'] / 100
    r['temp_delta_engine_coolant'] = r['engine_temp_c'] - r['coolant_temp_c']
    r['oil_pressure_per_rpm']      = r['oil_pressure_psi'] / (r['engine_rpm'] + 1)
    wheel_cols = ['wheel_speed_fl_kph','wheel_speed_fr_kph',
                  'wheel_speed_rl_kph','wheel_speed_rr_kph']
    r['wheel_speed_mean']   = row[wheel_cols].mean()
    r['wheel_speed_std']    = row[wheel_cols].std()
    r['brake_stress_event'] = r['vehicle_speed_kph'] * r['brake_pedal_pos_percent'] / 100
    r['alternator_deficit'] = r['battery_voltage_v'] - r['alternator_output_v']
    r['charge_efficiency']  = r['battery_charge_percent'] * r['battery_health_percent'] / 100
    for col in ['engine_temp_c','engine_rpm','oil_pressure_psi','coolant_temp_c',
                'exhaust_gas_temp_c','brake_temp_c','brake_pad_wear_mm',
                'brake_fluid_level_psi','battery_voltage_v','battery_charge_percent',
                'battery_health_percent','alternator_output_v']:
        r[f'{col}_roll_mean'] = r[col]
        r[f'{col}_roll_std']  = 0.0
    return r

# ── Subsystem IF scoring ──────────────────────────────────────────────────────
def subsystem_status(eng_row, if_models):
    status = {}
    for system in ['Engine', 'Brake', 'Battery']:
        feats  = if_models[system]['features']
        scaler = if_models[system]['scaler']
        iso    = if_models[system]['model']
        X      = scaler.transform(pd.DataFrame([eng_row[feats]]))
        score  = iso.decision_function(X)[0]
        flag   = iso.predict(X)[0]
        status[system] = {'score': round(score, 3), 'anomalous': flag == -1}
    return status

# ── SHAP top signals ──────────────────────────────────────────────────────────
def top_shap(X_row, xgb_model, feature_cols, n=5):
    explainer = shap.TreeExplainer(xgb_model)
    vals      = explainer.shap_values(X_row[feature_cols])[0]
    series    = pd.Series(vals, index=feature_cols)
    top       = series.abs().nlargest(n).index
    return [(f, round(X_row[f].iloc[0], 3), round(series[f], 4)) for f in top]

# ── Print report ──────────────────────────────────────────────────────────────
def print_report(sensor_input: dict, prob: float,
                  sub_status: dict, shap_signals: list,
                  preset_name: str, overrides: dict):
    W = 64
    if prob >= 0.80:   level, rec = '⚠️  HIGH RISK',   'Immediate inspection required'
    elif prob >= 0.40: level, rec = '🔶 MEDIUM RISK', 'Schedule maintenance within 48h'
    else:              level, rec = '✅ LOW RISK',    'Continue normal monitoring'

    print()
    print('╔' + '═' * W + '╗')
    print('║' + 'SENSOR-BASED FAILURE RISK ASSESSMENT'.center(W) + '║')
    print('╚' + '═' * W + '╝')
    print(f'  Preset   : {preset_name.upper()}')
    if overrides:
        print(f'  Overrides: {overrides}')
    print()
    print(f'  FAILURE RISK SCORE : {prob:.4f}')
    bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
    print(f'  [{bar}]  {level}')
    print()

    print('  SUBSYSTEM STATUS')
    for system, info in sub_status.items():
        icon  = '🔴' if info['anomalous'] else '🟢'
        state = 'ANOMALOUS' if info['anomalous'] else 'Normal   '
        print(f'    {icon} {system:8s}: {state}  (IF score: {info["score"]:+.3f})')
    print()

    print('  TOP SIGNALS DRIVING PREDICTION  (SHAP values)')
    for feat, val, shap_val in shap_signals:
        direction = '↑ pushes toward FAILURE' if shap_val > 0 else '↓ pushes toward NORMAL'
        # Flag out-of-range values
        in_range = ''
        if feat in NORMAL_RANGES:
            lo, hi = NORMAL_RANGES[feat]
            if val < lo: in_range = f'  ⚠ below normal ({lo}–{hi})'
            elif val > hi: in_range = f'  ⚠ above normal ({lo}–{hi})'
        print(f'    {feat:35s} = {val}')
        print(f'    {"":35s}   {direction} ({shap_val:+.4f}){in_range}')
    print()

    print(f'  RECOMMENDATION : {rec}')
    print('─' * (W + 2))
    print()

    # Show which sensors are outside normal range
    flagged = []
    for sensor, (lo, hi) in NORMAL_RANGES.items():
        val = sensor_input.get(sensor)
        if val is not None and (val < lo or val > hi):
            direction = 'LOW' if val < lo else 'HIGH'
            flagged.append((sensor, val, lo, hi, direction))

    if flagged:
        print('  SENSORS OUTSIDE NORMAL RANGE')
        for sensor, val, lo, hi, direction in flagged:
            print(f'    ⚠  {sensor:35s}: {val}  '
                  f'({direction}, normal: {lo}–{hi})')
        print()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='normal',
                        choices=list(PRESETS.keys()),
                        help='Sensor scenario preset')
    parser.add_argument('--set',   nargs='+', default=[],
                        metavar='sensor=value',
                        help='Override individual sensors, e.g. --set engine_temp_c=130')
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for item in args.set:
        try:
            k, v = item.split('=')
            overrides[k.strip()] = float(v.strip())
        except ValueError:
            print(f'Bad override format: {item}  (expected sensor=value)')
            sys.exit(1)

    # Build sensor input: baseline + preset + overrides
    sensor_input = {**NORMAL_BASELINE, **PRESETS[args.preset], **overrides}

    # Validate all required sensors present
    missing = [k for k in NORMAL_BASELINE if k not in sensor_input]
    if missing:
        print(f'Missing sensors: {missing}')
        sys.exit(1)

    # Load models
    xgb_model    = joblib.load(XGB_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    if_models    = joblib.load(IF_PATH)

    # Engineer features
    row = pd.Series(sensor_input)
    eng = engineer(row)
    X   = pd.DataFrame([eng])

    # Predict
    prob      = xgb_model.predict_proba(X[feature_cols])[0][1]
    sub_stat  = subsystem_status(eng, if_models)
    shap_sigs = top_shap(X, xgb_model, feature_cols, n=5)

    print_report(sensor_input, prob, sub_stat, shap_sigs,
                  args.preset, overrides)

if __name__ == '__main__':
    main()