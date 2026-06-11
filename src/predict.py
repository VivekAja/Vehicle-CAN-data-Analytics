"""
Vehicle Predictive Maintenance — Inference Pipeline
----------------------------------------------------
Produces one master health report per vehicle, aggregating
all sensor readings into a single status output.

Usage:
    python src/predict.py --vehicle VEH0004   # single vehicle report
    python src/predict.py --all               # report for every vehicle
    python src/predict.py --risk high         # only HIGH risk vehicles
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE, '..', 'outputs', 'models')
DATA_PATH   = os.path.join(BASE, '..', 'data', 'df_anomaly.csv')

XGB_PATH          = os.path.join(MODEL_DIR, 'xgb_any_failure.pkl')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.pkl')
IF_MODELS_PATH    = os.path.join(MODEL_DIR, 'isolation_forests.pkl')

# ── Feature engineering (mirrors NB02) ────────────────────────────────────────
def engineer_features(row: pd.Series) -> pd.Series:
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
    rolling_signals = [
        'engine_temp_c','engine_rpm','oil_pressure_psi','coolant_temp_c',
        'exhaust_gas_temp_c','brake_temp_c','brake_pad_wear_mm',
        'brake_fluid_level_psi','battery_voltage_v','battery_charge_percent',
        'battery_health_percent','alternator_output_v'
    ]
    for col in rolling_signals:
        r[f'{col}_roll_mean'] = r[col]
        r[f'{col}_roll_std']  = 0.0
    return r

# ── Score all rows for a vehicle ──────────────────────────────────────────────
def score_vehicle_rows(vdf: pd.DataFrame,
                        xgb_model,
                        if_models: dict,
                        feature_cols: list) -> pd.DataFrame:
    """Run inference on every row, return scores + subsystem flags."""
    records = []
    feature_sets = {
        'Engine': if_models['Engine']['features'],
        'Brake':  if_models['Brake']['features'],
        'Battery':if_models['Battery']['features'],
    }

    for _, row in vdf.iterrows():
        eng   = engineer_features(row)
        X_row = pd.DataFrame([eng])
        prob  = xgb_model.predict_proba(X_row[feature_cols])[0][1]

        sub_flags = {}
        for system, feats in feature_sets.items():
            scaler = if_models[system]['scaler']
            iso    = if_models[system]['model']
            X_if   = scaler.transform(pd.DataFrame([eng[feats]]))
            score  = iso.decision_function(X_if)[0]
            flag   = iso.predict(X_if)[0]
            sub_flags[f'{system.lower()}_if_score'] = round(score, 3)
            sub_flags[f'{system.lower()}_anomalous'] = (flag == -1)

        records.append({
            'timestamp':    row['timestamp'],
            'failure_prob': round(prob, 4),
            'actual_label': row.get('any_failure', np.nan),
            **sub_flags
        })

    return pd.DataFrame(records)

# ── Sparkline from probabilities ──────────────────────────────────────────────
def sparkline(probs: list, width: int = 10) -> str:
    bars = ' ▁▂▃▄▅▆▇█'
    recent = probs[-width:]
    return ''.join(bars[min(int(p * 8), 8)] for p in recent)

# ── Risk level ────────────────────────────────────────────────────────────────
def risk_label(prob: float) -> tuple:
    if prob >= 0.80: return '⚠️  HIGH RISK',   'Immediate inspection required'
    if prob >= 0.40: return '🔶 MEDIUM RISK', 'Schedule maintenance within 48h'
    return              '✅ LOW RISK',    'Continue normal monitoring'

# ── Master vehicle report ─────────────────────────────────────────────────────
def vehicle_report(vehicle_id: str,
                   df_full: pd.DataFrame,
                   xgb_model,
                   if_models: dict,
                   feature_cols: list):

    vdf = df_full[df_full['vehicle_id'] == vehicle_id].sort_values('timestamp')
    if vdf.empty:
        print(f'No records found for {vehicle_id}')
        return None

    brand      = vdf['brand'].iloc[0]
    t_start    = pd.to_datetime(vdf['timestamp'].iloc[0]).strftime('%Y-%m-%d %H:%M')
    t_end      = pd.to_datetime(vdf['timestamp'].iloc[-1]).strftime('%Y-%m-%d %H:%M')
    n_records  = len(vdf)

    # Score all rows
    scored = score_vehicle_rows(vdf, xgb_model, if_models, feature_cols)

    # Aggregate metrics
    peak_prob    = scored['failure_prob'].max()
    latest_prob  = scored['failure_prob'].iloc[-1]
    n_high_risk  = (scored['failure_prob'] >= 0.80).sum()

    # Subsystem health
    sub_summary = {}
    for system in ['engine', 'brake', 'battery']:
        n_anom  = scored[f'{system}_anomalous'].sum()
        avg_score = scored[f'{system}_if_score'].mean()
        sub_summary[system] = {'n_anom': n_anom, 'avg_score': avg_score}

    # Top normal ranges from the full dataset for context
    normal_df = df_full[df_full['any_failure'] == 0]
    ranges = {
        'brake_fluid_level_psi': (normal_df['brake_fluid_level_psi'].quantile(0.05),
                                   normal_df['brake_fluid_level_psi'].quantile(0.95)),
        'brake_temp_c':          (normal_df['brake_temp_c'].quantile(0.05),
                                   normal_df['brake_temp_c'].quantile(0.95)),
        'battery_voltage_v':     (normal_df['battery_voltage_v'].quantile(0.05),
                                   normal_df['battery_voltage_v'].quantile(0.95)),
    }

    # Top 2 failure readings by prob
    top_failure_rows = vdf.loc[scored.nlargest(2, 'failure_prob').index]

    # Risk for overall report based on peak
    level, recommendation = risk_label(peak_prob)

    # ── Print ────────────────────────────────────────────────────────────────
    W = 64
    print('╔' + '═' * W + '╗')
    print('║' + 'VEHICLE HEALTH REPORT'.center(W) + '║')
    print('╚' + '═' * W + '╝')
    print(f'  Vehicle : {vehicle_id}  │  Brand: {brand}')
    print(f'  Period  : {t_start} → {t_end}')
    print(f'  Records : {n_records} readings analysed')
    print()
    print(f'  CURRENT STATUS  →  {level}')
    print(f'  Peak risk score : {peak_prob:.2f}  │  '
          f'Latest: {latest_prob:.2f}  │  '
          f'High-risk readings: {n_high_risk}/{n_records}')
    print()

    print('  SUBSYSTEM HEALTH')
    icons  = {'engine': '🔴' if sub_summary['engine']['n_anom']  > 0 else '🟢',
               'brake':  '🔴' if sub_summary['brake']['n_anom']   > 0 else '🟢',
               'battery':'🔴' if sub_summary['battery']['n_anom'] > 0 else '🟢'}
    labels = {'engine': 'Engine ', 'brake': 'Brake  ', 'battery': 'Battery'}
    key_signals = {
        'engine':  ('engine_temp_c',          'Avg temp',  '°C'),
        'brake':   ('brake_pad_wear_mm',       'Avg wear',  'mm'),
        'battery': ('battery_charge_percent',  'Avg charge','%'),
    }
    for sys in ['engine', 'brake', 'battery']:
        n     = sub_summary[sys]['n_anom']
        state = 'AT RISK ' if n > 0 else 'Healthy '
        col, lbl, unit = key_signals[sys]
        avg_val = vdf[col].mean()
        print(f"    {icons[sys]} {labels[sys]}: {state} │ "
              f"{n} anomalous reading(s) │ {lbl}: {avg_val:.1f}{unit}")
    print()

    print('  RISK TIMELINE  (all readings, newest last)')
    probs = scored['failure_prob'].tolist()
    times = pd.to_datetime(vdf['timestamp']).dt.strftime('%H:%M').tolist()
    # Print up to last 12 readings
    show = list(zip(times, probs))[-12:]
    for t, p in show:
        bar   = '█' if p >= 0.80 else ('▄' if p >= 0.40 else '▁')
        flag  = '  ← FAILURE DETECTED' if p >= 0.80 else ''
        print(f'    {t}  {bar}  {p:.2f}{flag}')
    print()

    if n_high_risk > 0:
        print('  TOP FAILURE SIGNALS  (from highest-risk reading)')
        failure_row = vdf.loc[scored['failure_prob'].idxmax()]
        checks = [
            ('brake_fluid_level_psi', ranges['brake_fluid_level_psi'],
             f"{failure_row['brake_fluid_level_psi']:.1f} PSI",
             f"normal: {ranges['brake_fluid_level_psi'][0]:.0f}–{ranges['brake_fluid_level_psi'][1]:.0f}"),
            ('brake_temp_c', ranges['brake_temp_c'],
             f"{failure_row['brake_temp_c']:.1f}°C",
             f"normal: {ranges['brake_temp_c'][0]:.0f}–{ranges['brake_temp_c'][1]:.0f}"),
            ('battery_voltage_v', ranges['battery_voltage_v'],
             f"{failure_row['battery_voltage_v']:.2f}V",
             f"normal: {ranges['battery_voltage_v'][0]:.1f}–{ranges['battery_voltage_v'][1]:.1f}"),
        ]
        i = 1
        for col, (lo, hi), val_str, range_str in checks:
            actual = failure_row[col]
            if actual < lo or actual > hi:
                direction = 'dropped to' if actual < lo else 'spiked to'
                print(f'    {i}. {col:30s} {direction} {val_str}  ({range_str})')
                i += 1
        if i == 1:
            print('    (signals within normal range — multi-feature combination triggered alert)')
        print()

    print(f'  RECOMMENDATION : {recommendation}')
    print('─' * (W + 2))
    print()

    return {
        'vehicle_id':   vehicle_id,
        'brand':        brand,
        'peak_prob':    peak_prob,
        'latest_prob':  latest_prob,
        'n_high_risk':  n_high_risk,
        'risk_level':   level.split()[1],
    }

# ── Fleet summary table ───────────────────────────────────────────────────────
def fleet_summary(results: list):
    rdf = pd.DataFrame(results).sort_values('peak_prob', ascending=False)
    high   = rdf[rdf['risk_level'] == 'RISK,']  # won't match cleanly, use prob
    high   = rdf[rdf['peak_prob'] >= 0.80]
    medium = rdf[(rdf['peak_prob'] >= 0.40) & (rdf['peak_prob'] < 0.80)]
    low    = rdf[rdf['peak_prob'] < 0.40]

    print('╔' + '═' * 64 + '╗')
    print('║' + 'FLEET RISK SUMMARY'.center(64) + '║')
    print('╚' + '═' * 64 + '╝')
    print(f'  Total vehicles : {len(rdf)}')
    print(f'  🔴 HIGH RISK   : {len(high)}  vehicles')
    print(f'  🔶 MEDIUM RISK : {len(medium)} vehicles')
    print(f'  🟢 LOW RISK    : {len(low)}  vehicles')
    print()
    if len(high) > 0:
        print('  HIGH RISK VEHICLES — inspect immediately:')
        for _, row in high.iterrows():
            print(f"    {row['vehicle_id']:10s} ({row['brand']:12s})  "
                  f"peak score: {row['peak_prob']:.2f}  │  "
                  f"high-risk readings: {row['n_high_risk']}")
    print('─' * 66)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle', type=str,   default=None)
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--risk',    type=str,   default=None,
                        choices=['high', 'medium', 'low'])
    args = parser.parse_args()

    print('Loading models...')
    xgb_model    = joblib.load(XGB_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    if_models    = joblib.load(IF_MODELS_PATH)
    df           = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    print(f'  {df.vehicle_id.nunique()} vehicles loaded\n')

    if args.vehicle:
        vehicle_report(args.vehicle, df, xgb_model, if_models, feature_cols)

    elif args.all or args.risk:
        vehicles = df['vehicle_id'].unique()
        results  = []
        for vid in vehicles:
            r = vehicle_report(vid, df, xgb_model, if_models, feature_cols)
            if r:
                results.append(r)
        fleet_summary(results)

    else:
        # Default: show 3 most at-risk vehicles
        print('No flag given — showing top 3 highest-risk vehicles.\n')
        print('Usage:')
        print('  --vehicle VEH0004   single vehicle report')
        print('  --all               all 50 vehicles + fleet summary')
        print('  --risk high         only HIGH risk vehicles\n')

        vehicles = df['vehicle_id'].unique()
        results  = []
        for vid in vehicles:
            vdf   = df[df['vehicle_id'] == vid]
            eng   = vdf.apply(engineer_features, axis=1)
            probs = [xgb_model.predict_proba(
                        pd.DataFrame([e])[feature_cols])[0][1]
                     for _, e in eng.iterrows()]
            results.append({'vehicle_id': vid, 'peak_prob': max(probs),
                             'brand': vdf['brand'].iloc[0]})

        top3 = sorted(results, key=lambda x: x['peak_prob'], reverse=True)[:3]
        for r in top3:
            vehicle_report(r['vehicle_id'], df, xgb_model, if_models, feature_cols)

if __name__ == '__main__':
    main()