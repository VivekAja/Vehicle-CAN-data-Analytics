"""
Vehicle Predictive Maintenance — Sensor Simulation
---------------------------------------------------
Two-layer prediction:
  Layer 1 — Rule-based hard limits  (physics thresholds, always checked)
  Layer 2 — XGBoost ML model        (learned multi-signal patterns)

Final risk = max(rule_risk, ml_risk)

Usage:
    python src/simulate.py --preset normal
    python src/simulate.py --preset brake_wear
    python src/simulate.py --preset engine_stress
    python src/simulate.py --preset battery_drain
    python src/simulate.py --preset critical
    python src/simulate.py --preset normal --set coolant_temp_c=300
    python src/simulate.py --preset normal --set brake_fluid_level_psi=200
"""

import pandas as pd
import numpy as np
import joblib
import shap
import argparse
import sys
import os

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, '..', 'outputs', 'models')
XGB_PATH  = os.path.join(MODEL_DIR, 'xgb_any_failure.pkl')
FEAT_PATH = os.path.join(MODEL_DIR, 'feature_cols.pkl')
IF_PATH   = os.path.join(MODEL_DIR, 'isolation_forests.pkl')

# ── Normal operating baseline ────────────────────────────────────────────────
NORMAL_BASELINE = {
    'engine_temp_c': 88.0, 'engine_rpm': 2100.0, 'oil_pressure_psi': 55.0,
    'coolant_temp_c': 85.0, 'fuel_level_percent': 65.0,
    'fuel_consumption_lph': 5.2, 'engine_load_percent': 48.0,
    'throttle_pos_percent': 35.0, 'air_flow_rate_gps': 18.0,
    'exhaust_gas_temp_c': 320.0, 'vibration_level': 1.8, 'engine_hours': 850.0,
    'brake_fluid_level_psi': 900.0, 'brake_pad_wear_mm': 8.0,
    'brake_temp_c': 120.0, 'abs_fault_indicator': 0,
    'brake_pedal_pos_percent': 15.0,
    'wheel_speed_fl_kph': 80.0, 'wheel_speed_fr_kph': 80.0,
    'wheel_speed_rl_kph': 80.0, 'wheel_speed_rr_kph': 80.0,
    'battery_voltage_v': 12.8, 'battery_current_a': 14.0,
    'battery_temp_c': 28.0, 'alternator_output_v': 14.2,
    'battery_charge_percent': 78.0, 'battery_health_percent': 88.0,
    'vehicle_speed_kph': 80.0, 'ambient_temp_c': 22.0,
    'humidity_percent': 55.0, 'odometer_reading': 42000.0,
}

PRESETS = {
    'normal': {},
    'brake_wear': {
        'brake_fluid_level_psi': 320.0, 'brake_pad_wear_mm': 1.8,
        'brake_temp_c': 290.0, 'abs_fault_indicator': 1,
    },
    'engine_stress': {
        'engine_temp_c': 128.0, 'oil_pressure_psi': 16.0,
        'vibration_level': 5.2, 'exhaust_gas_temp_c': 680.0,
        'engine_rpm': 4800.0, 'coolant_temp_c': 118.0,
    },
    'battery_drain': {
        'battery_voltage_v': 10.9, 'battery_charge_percent': 12.0,
        'battery_health_percent': 42.0, 'alternator_output_v': 11.1,
    },
    'critical': {
        'brake_fluid_level_psi': 280.0, 'brake_temp_c': 310.0,
        'engine_temp_c': 132.0, 'oil_pressure_psi': 12.0,
        'battery_voltage_v': 10.8, 'battery_health_percent': 35.0,
        'vibration_level': 6.1, 'abs_fault_indicator': 1,
        'coolant_temp_c': 125.0,
    },
}

# ── Layer 1: Hard rule thresholds ─────────────────────────────────────────────
# Each rule: (sensor, operator, threshold, severity, subsystem, description)
# severity: 'critical' → prob=1.0 override, 'warning' → prob=0.85
RULES = [
    # Engine
    ('engine_temp_c',          '>',  115, 'critical', 'Engine',  'Engine overheating'),
    ('coolant_temp_c',         '>',  110, 'critical', 'Engine',  'Coolant overheating — cooling system failure'),
    ('oil_pressure_psi',       '<',   20, 'critical', 'Engine',  'Critically low oil pressure'),
    ('oil_pressure_psi',       '<',   30, 'warning',  'Engine',  'Low oil pressure'),
    ('vibration_level',        '>',   4.5,'warning',  'Engine',  'Abnormal vibration'),
    ('exhaust_gas_temp_c',     '>',   600,'warning',  'Engine',  'Exhaust temp elevated'),
    # Brake
    ('brake_fluid_level_psi',  '<',  350, 'critical', 'Brake',   'Critically low brake fluid'),
    ('brake_fluid_level_psi',  '<',  500, 'warning',  'Brake',   'Low brake fluid pressure'),
    ('brake_temp_c',           '>',  260, 'critical', 'Brake',   'Brake thermal overload'),
    ('brake_pad_wear_mm',      '<',    2, 'critical', 'Brake',   'Brake pads critically worn'),
    ('brake_pad_wear_mm',      '<',    4, 'warning',  'Brake',   'Brake pads low'),
    ('abs_fault_indicator',    '==',   1, 'warning',  'Brake',   'ABS fault active'),
    # Battery
    ('battery_voltage_v',      '<',  11.5,'critical', 'Battery', 'Battery voltage critically low'),
    ('battery_voltage_v',      '<',  12.0,'warning',  'Battery', 'Battery voltage low'),
    ('battery_health_percent', '<',   50, 'critical', 'Battery', 'Battery health critically degraded'),
    ('battery_health_percent', '<',   65, 'warning',  'Battery', 'Battery health degraded'),
    ('battery_charge_percent', '<',   15, 'critical', 'Battery', 'Battery almost discharged'),
]

def check_rules(sensors: dict) -> list:
    """Return list of triggered rules with their severity."""
    triggered = []
    for sensor, op, threshold, severity, subsystem, desc in RULES:
        val = sensors.get(sensor, None)
        if val is None:
            continue
        fired = False
        if   op == '>'  and val >  threshold: fired = True
        elif op == '<'  and val <  threshold: fired = True
        elif op == '>=' and val >= threshold: fired = True
        elif op == '<=' and val <= threshold: fired = True
        elif op == '==' and val == threshold: fired = True
        if fired:
            triggered.append({
                'sensor': sensor, 'value': val,
                'threshold': threshold, 'op': op,
                'severity': severity, 'subsystem': subsystem,
                'description': desc
            })
    return triggered

def rule_risk_score(triggered: list) -> float:
    """Convert rule violations to a 0–1 risk score."""
    if not triggered:
        return 0.0
    if any(r['severity'] == 'critical' for r in triggered):
        return 1.0
    return 0.85   # warning-only

# ── Feature engineering ───────────────────────────────────────────────────────
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

def top_shap(X_row, xgb_model, feature_cols, n=5):
    explainer = shap.TreeExplainer(xgb_model)
    vals      = explainer.shap_values(X_row[feature_cols])[0]
    series    = pd.Series(vals, index=feature_cols)
    top       = series.abs().nlargest(n).index
    return [(f, round(X_row[f].iloc[0], 3), round(series[f], 4)) for f in top]

# ── Print report ──────────────────────────────────────────────────────────────
def print_report(sensors, ml_prob, rule_prob, final_prob,
                  triggered_rules, sub_status, shap_signals,
                  preset_name, overrides):
    W = 64

    if final_prob >= 0.80:   level, rec = '⚠️  HIGH RISK',   'Immediate inspection required'
    elif final_prob >= 0.40: level, rec = '🔶 MEDIUM RISK', 'Schedule maintenance within 48h'
    else:                    level, rec = '✅ LOW RISK',    'Continue normal monitoring'

    print()
    print('╔' + '═' * W + '╗')
    print('║' + 'SENSOR-BASED FAILURE RISK ASSESSMENT'.center(W) + '║')
    print('╚' + '═' * W + '╝')
    print(f'  Preset   : {preset_name.upper()}')
    if overrides:
        for k, v in overrides.items():
            print(f'  Override : {k} = {v}')
    print()

    # Risk scores — show both layers
    bar_ml    = '█' * int(ml_prob    * 30) + '░' * (30 - int(ml_prob    * 30))
    bar_rule  = '█' * int(rule_prob  * 30) + '░' * (30 - int(rule_prob  * 30))
    bar_final = '█' * int(final_prob * 30) + '░' * (30 - int(final_prob * 30))

    print('  RISK SCORES')
    print(f'  ML Model   [{bar_ml}]  {ml_prob:.4f}')
    print(f'  Rule Layer [{bar_rule}]  {rule_prob:.4f}')
    print(f'  ─────────────────────────────────────────────')
    print(f'  FINAL RISK [{bar_final}]  {final_prob:.4f}  →  {level}')
    print()

    # Rule violations
    if triggered_rules:
        print('  RULE VIOLATIONS  (physics thresholds)')
        for r in triggered_rules:
            icon = '🔴' if r['severity'] == 'critical' else '🟡'
            print(f"    {icon} [{r['subsystem']:7s}] {r['description']}")
            print(f"       {r['sensor']} = {r['value']}  "
                  f"(threshold: {r['op']} {r['threshold']})")
        print()

    # Subsystem status
    print('  SUBSYSTEM STATUS  (Isolation Forest)')
    for system, info in sub_status.items():
        icon  = '🔴' if info['anomalous'] else '🟢'
        state = 'ANOMALOUS' if info['anomalous'] else 'Normal   '
        print(f'    {icon} {system:8s}: {state}  (IF score: {info["score"]:+.3f})')
    print()

    # SHAP signals from ML
    print('  TOP ML SIGNALS  (SHAP — what XGBoost learned)')
    for feat, val, shap_val in shap_signals:
        direction = '↑ toward FAILURE' if shap_val > 0 else '↓ toward NORMAL'
        print(f'    {feat:35s} = {val:<10}  {direction} ({shap_val:+.4f})')
    print()

    print(f'  RECOMMENDATION : {rec}')
    print('─' * (W + 2))

    # Explain the two-layer design
    if triggered_rules and ml_prob < 0.5:
        print()
        print('  ℹ  NOTE: ML score is low but rules fired.')
        print('     This sensor pattern was not in the training labels.')
        print('     Rule-based layer correctly overrides the ML model here.')
        print('     Real systems always combine ML + hard safety limits.')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='normal',
                        choices=list(PRESETS.keys()))
    parser.add_argument('--set', nargs='+', default=[], metavar='sensor=value')
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        try:
            k, v = item.split('=')
            overrides[k.strip()] = float(v.strip())
        except ValueError:
            print(f'Bad format: {item}  (use sensor=value)')
            sys.exit(1)

    sensors = {**NORMAL_BASELINE, **PRESETS[args.preset], **overrides}

    xgb_model    = joblib.load(XGB_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    if_models    = joblib.load(IF_PATH)

    row  = pd.Series(sensors)
    eng  = engineer(row)
    X    = pd.DataFrame([eng])

    # Layer 1 — rules
    triggered  = check_rules(sensors)
    rule_prob  = rule_risk_score(triggered)

    # Layer 2 — ML
    ml_prob    = xgb_model.predict_proba(X[feature_cols])[0][1]

    # Final — take max
    final_prob = max(ml_prob, rule_prob)

    sub_stat   = subsystem_status(eng, if_models)
    shap_sigs  = top_shap(X, xgb_model, feature_cols, n=5)

    print_report(sensors, ml_prob, rule_prob, final_prob,
                  triggered, sub_stat, shap_sigs,
                  args.preset, overrides)

if __name__ == '__main__':
    main()
