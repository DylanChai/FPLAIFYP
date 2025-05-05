import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from train_goalsTeamStrength import enhance_with_team_data, prepare_gw_data

def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA = pd.read_csv(ROOT/'data/processed/merged_gw_cleaned.csv', on_bad_lines='skip')
    TEAMS = pd.read_csv(ROOT/'data/processed/teams.csv')
    FIX   = pd.read_csv(ROOT/'data/processed/fixtures.csv', parse_dates=['kickoff_time'])

    # latest completed GW (same logic as goals model)
    now = pd.Timestamp.now(tz='UTC')
    latest_c = int(FIX[FIX.kickoff_time <= now]['event'].max())
    target_gw = latest_c + 1

    #) TRAIN SET  (GW 1-32) ────────────────────────────────
    train = DATA[DATA.GW.between(1, min(latest_c, 32))].copy()
    for col in ['expected_assists', 'creativity', 'minutes']:
        train[f'roll3_{col}'] = (train.groupby('name')[col]
                                   .transform(lambda x: x.rolling(3, 1).mean()))
    train = enhance_with_team_data(train, TEAMS)

    # role dummies
    train['is_forward']   = (train.position == 'FWD').astype(int)
    train['is_midfielder']= (train.position == 'MID').astype(int)
    train['is_defender']  = (train.position == 'DEF').astype(int)

    FEATURES = [
        'roll3_minutes', 'roll3_creativity', 'expected_assists',
        'ict_index', 'influence', 'transfers_in',
        'team_attacking', 'opp_defending', 'fixture_difficulty',
        'is_forward', 'is_midfielder', 'is_defender'
    ]
    TARGET = 'assists'

    # numeric + impute for example players who came in January to average their result for NAN results 
    train[FEATURES+[TARGET]] = train[FEATURES+[TARGET]].apply(pd.to_numeric, errors='coerce')
    train = train.dropna(subset=[TARGET])
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(train[FEATURES])
    y = train[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                               max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, rf.predict(X_te))
    print(f"MAE: {mae:.5f}")

    # ── NEW: Feature importance ───────────────────────────────
    importances = pd.DataFrame({
        'feature': FEATURES,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    fi_path = ROOT / "models" / "assists_feature_importance.csv"
    importances.to_csv(fi_path, index=False)
    print(f"✅ Feature importance saved to {fi_path.name}")

    # ── 2) PREDICTION SET  (upcoming GW) ───────────────────────
    players  = train[['name','team','position']].drop_duplicates()
    num_cols = train.select_dtypes(np.number).columns
    roll_avg = train.groupby(['name','team'])[num_cols].mean().reset_index()
    gw_df    = prepare_gw_data(players, roll_avg, TEAMS, FIX, target_gw)

    # drop blank-GW and cold players GW34 teams like Arsenal 
    gw_df = gw_df[gw_df.opponent_name != 'Unknown']
    if 'roll3_minutes' not in gw_df.columns:
        gw_df['roll3_minutes'] = gw_df['minutes']
    gw_df = gw_df[gw_df['roll3_minutes'] >= 30]

    # role dummies again
    gw_df['is_forward']   = (gw_df.position == 'FWD').astype(int)
    gw_df['is_midfielder']= (gw_df.position == 'MID').astype(int)
    gw_df['is_defender']  = (gw_df.position == 'DEF').astype(int)

    X_future = imputer.transform(gw_df[FEATURES])
    probs    = rf.predict(X_future).clip(0).round(2)
    gw_df['predicted_assists'] = probs

    # save
    out_cols = ['name','team','position','opponent_name','was_home',
                'fixture_difficulty','predicted_assists']
    fname = ROOT/f"models/GW{target_gw}_Predicted_assists.csv"
    gw_df.sort_values('predicted_assists', ascending=False)[out_cols].to_csv(fname, index=False)
    print("✅ Saved", fname.name)

if __name__ == "__main__":
    main()
