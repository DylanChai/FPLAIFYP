# models/train_cleansheetsTeamStrength.py
# ───────────────────────────────────────────────────────────────────────────
# Clean‑sheet probability model (GK/DEF) – calibrated & balanced
# Author: Dylan Chai — May 2025
# ───────────────────────────────────────────────────────────────────────────
"""
Key points
──────────
• Uses team‑average xGC, team/opponent strength (z‑scored) and baseline CS rates.
• Adds `opp_cs_allow` – how often the opponent *gives up* clean sheets.
• Random‑Forest (balanced depth/leaves) → logistic (Platt) calibration.
• Caps final probabilities at 0.50 to match betting‑market ceilings.
Outputs
───────
• models/cleansheet_feature_importance.csv
• models/GW<next>_Predicted_clean_sheets.csv
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ── Helpers ───────────────────────────────────────────────────────────────

def enhance_with_team_data(df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    name2id = {r['name']: r['id'] for _, r in teams_df.iterrows()}
    strength = {
        r['id']:{
            'def_home': r['strength_defence_home'],
            'def_away': r['strength_defence_away'],
            'atk_home': r['strength_attack_home'],
            'atk_away': r['strength_attack_away'],
            'overall_home': r['strength_overall_home'],
            'overall_away': r['strength_overall_away'],
        } for _, r in teams_df.iterrows()
    }
    out = df.copy()
    out['team_def'] = out.apply(lambda r: strength.get(name2id.get(r['team'],0),{})
                                             .get('def_home' if r['was_home'] else 'def_away',1100), axis=1)
    out['opp_att']  = out.apply(lambda r: strength.get(r['opponent_team'],{})
                                             .get('atk_away' if r['was_home'] else 'atk_home',1100), axis=1)
    out['team_overall'] = out.apply(lambda r: strength.get(name2id.get(r['team'],0),{})
                                                  .get('overall_home' if r['was_home'] else 'overall_away',1100), axis=1)
    out['opp_overall']  = out.apply(lambda r: strength.get(r['opponent_team'],{})
                                                  .get('overall_home' if r['was_home'] else 'overall_away',1100), axis=1)
    out['fixture_difficulty'] = ((out['team_def']-out['opp_att']) + (out['team_overall']-out['opp_overall']))/100.0
    return out


def attach_fixture_info(df: pd.DataFrame, teams_df: pd.DataFrame, fixtures_df: pd.DataFrame, gw: int) -> pd.DataFrame:
    id2name = {r['id']: r['name'] for _, r in teams_df.iterrows()}
    name2id = {v:k for k,v in id2name.items()}
    fx = fixtures_df[fixtures_df['event']==gw]

    def lookup(row):
        tid = name2id.get(row['team'])
        if tid is None:
            return pd.Series([np.nan, False])
        m = fx[(fx['team_h']==tid)|(fx['team_a']==tid)]
        if m.empty:
            return pd.Series([np.nan, False])
        m = m.iloc[0]
        return pd.Series([m['team_a'] if m['team_h']==tid else m['team_h'], m['team_h']==tid])

    df[['opponent_team','was_home']] = df.apply(lookup, axis=1)
    return df

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    BASE = Path(__file__).resolve().parents[1]
    data_csv, teams_csv, fixtures_csv = (
        BASE/'data'/'processed'/'merged_gw_cleaned.csv',
        BASE/'data'/'processed'/'teams.csv',
        BASE/'data'/'processed'/'fixtures.csv'
    )
    out_pred = BASE/'models'/f'GW34_Predicted_clean_sheets.csv'
    out_imp  = BASE/'models'/'cleansheet_feature_importance.csv'

    df        = pd.read_csv(data_csv, low_memory=False, on_bad_lines='skip')
    teams_df  = pd.read_csv(teams_csv)
    fixtures  = pd.read_csv(fixtures_csv, parse_dates=['kickoff_time'])

    latest_done = int(fixtures.loc[fixtures['kickoff_time']<=pd.Timestamp.utcnow(),'event'].max())
    next_gw     = latest_done + 1

    # Filter GK/DEF ≥ 60 mins
    df = df[df['position'].isin(['GK','GKP','DEF']) & (df['minutes']>=60)].copy()

    # Training window GW1‑32
    train_df = df[df['GW'].between(1, min(latest_done,32))].copy()

    # Add strengths & rolling minutes
    train_df = enhance_with_team_data(train_df, teams_df)
    train_df['roll3_minutes'] = train_df.groupby('name')['minutes'].transform(lambda x: x.rolling(3,1).mean())

    # Baselines & opponent concession rate
    team_cs_avg  = train_df.groupby('team')['clean_sheets'].mean()
    opp_cs_allow = train_df.groupby('opponent_team')['clean_sheets'].mean()
    team_xgc_avg = train_df.groupby('team')['expected_goals_conceded'].mean()

    train_df['team_cs_avg']  = train_df['team'].map(team_cs_avg)
    train_df['opp_cs_allow'] = train_df['opponent_team'].map(opp_cs_allow)
    train_df['team_xgc']     = train_df['team'].map(team_xgc_avg)

    # Z‑scores
    z_cols = ['team_def','opp_att','team_xgc']
    for col in z_cols:
        m, s = train_df[col].mean(), train_df[col].std(ddof=0)
        train_df[f'{col}_z'] = (train_df[col]-m)/s
    fd_m, fd_s = train_df['fixture_difficulty'].mean(), train_df['fixture_difficulty'].std(ddof=0)
    train_df['fixture_diff_z'] = (train_df['fixture_difficulty']-fd_m)/fd_s

    features = [
        'roll3_minutes','team_def_z','opp_att_z','fixture_diff_z',
        'team_cs_avg','opp_cs_allow','team_xgc_z'
    ]
    target = 'clean_sheets'

    train_df[features+[target]] = train_df[features+[target]].apply(pd.to_numeric, errors='coerce')
    train_df.dropna(subset=[target], inplace=True)

    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(train_df[features])
    y = train_df[target]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    # --- Platt calibration ---
    val_raw = rf.predict(X_val).reshape(-1,1)
    platt = LogisticRegression(max_iter=200)
    platt.fit(val_raw, y_val)
    def calibrate(arr: np.ndarray) -> np.ndarray:
        return platt.predict_proba(arr.reshape(-1,1))[:,1]

    print(f'MAE (cal): {mean_absolute_error(y_val, calibrate(val_raw)):.4f}')

    # Save importances
    pd.DataFrame({'feature':features,'importance':rf.feature_importances_})\
      .sort_values('importance', ascending=False).to_csv(out_imp, index=False)

    # --- Upcoming GW prep ---
    players = df[['name','team']].drop_duplicates()
    numcols = train_df.select_dtypes(include=[np.number]).columns
    player_avg = train_df.groupby(['name','team'])[numcols].mean().reset_index()

    upcoming = players.merge(player_avg, on=['name','team'], how='left')
    upcoming = attach_fixture_info(upcoming, teams_df, fixtures, next_gw)
    upcoming.dropna(subset=['opponent_team'], inplace=True)
    upcoming = enhance_with_team_data(upcoming, teams_df)

    upcoming['roll3_minutes'] = upcoming['roll3_minutes'].fillna(upcoming['minutes'])
    upcoming['team_cs_avg']   = upcoming['team'].map(team_cs_avg).fillna(team_cs_avg.mean())
    upcoming['opp_cs_allow']  = upcoming['opponent_team'].map(opp_cs_allow).fillna(opp_cs_allow.mean())
    upcoming['team_xgc']      = upcoming['team'].map(team_xgc_avg).fillna(team_xgc_avg.mean())
    upcoming['fixture_diff_z']= (upcoming['fixture_difficulty']-fd_m)/fd_s

    for col in z_cols:
        m,s = train_df[col].mean(), train_df[col].std(ddof=0)
        upcoming[f'{col}_z'] = (upcoming[col]-m)/s

    X_fut = imp.transform(upcoming[features])
    upcoming['predicted_cs_prob'] = calibrate(rf.predict(X_fut)).clip(0,0.5).round(2)

    id2name = teams_df.set_index('id')['name'].to_dict()
    upcoming['opponent_name'] = upcoming['opponent_team'].map(id2name)

    cols_out = ['name','team','predicted_cs_prob','opponent_name','was_home','fixture_difficulty']
    upcoming.sort_values('predicted_cs_prob', ascending=False).to_csv(out_pred, columns=cols_out, index=False)
    print(f'Saved → {out_pred.name} (rows: {len(upcoming)})')
    print(upcoming[cols_out].head(15))

if __name__ == '__main__':
    main()
