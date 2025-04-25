"""
FPL Card-Probability Model  (yellow OR red in next GW)

Makes GW<xx>_Predicted_cards.csv with:
  name, team, position, opponent_name, was_home,
  fixture_difficulty, predicted_card_prob

Author: Dylan Chai
"""
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer

# reuse existing helpers
from train_goalsTeamStrength import enhance_with_team_data, prepare_gw_data

def main():
    ROOT = Path(__file__).resolve().parents[1]
    df   = pd.read_csv(ROOT/'data/processed/merged_gw_cleaned.csv', on_bad_lines='skip')
    teams= pd.read_csv(ROOT/'data/processed/teams.csv')
    fix  = pd.read_csv(ROOT/'data/processed/fixtures.csv', parse_dates=['kickoff_time'])

    # latest completed GW
    now  = pd.Timestamp.now(tz='UTC')
    last_played = int(fix[fix.kickoff_time <= now]['event'].max())
    pred_gw     = last_played + 1

    # ── 1. build training set (GW 1-32) ─────────────────────────
    tr = df[(df.GW.between(1, min(last_played, 32))) &
            (df.position != 'GK')].copy()

    for col in ['yellow_cards', 'red_cards', 'minutes']:
        tr[f'roll3_{col}'] = (tr.groupby('name')[col]
                                .transform(lambda x: x.rolling(3, 1).mean()))

    tr = enhance_with_team_data(tr, teams)

    tr['is_defender']   = (tr.position == 'DEF').astype(int)
    tr['is_midfielder'] = (tr.position == 'MID').astype(int)
    tr['is_forward']    = (tr.position == 'FWD').astype(int)

    FEATURES = [
        'roll3_minutes', 'roll3_yellow_cards', 'roll3_red_cards',
        'team_attacking', 'opp_defending', 'fixture_difficulty', 'was_home',
        'is_defender', 'is_midfielder', 'is_forward'
    ]
    tr[FEATURES] = tr[FEATURES].apply(pd.to_numeric, errors='coerce')

    tr['got_card'] = ((tr.yellow_cards + tr.red_cards) > 0).astype(int)

    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(tr[FEATURES])
    y = tr['got_card']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=20,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    print("Brier score:", round(brier_score_loss(y_te, rf.predict_proba(X_te)[:, 1]), 5))

    # ── 2. prepare upcoming GW dataframe ───────────────────────
    players  = tr[['name', 'team', 'position']].drop_duplicates()
    num_cols = tr.select_dtypes(np.number).columns
    roll_avg = tr.groupby(['name', 'team'])[num_cols].mean().reset_index()

    gw = prepare_gw_data(players, roll_avg, teams, fix, pred_gw)
    gw = gw[(gw.opponent_name != 'Unknown') & (gw.position != 'GK')]

    if 'roll3_minutes' not in gw.columns:
        gw['roll3_minutes'] = gw['minutes']
    gw = gw[gw.roll3_minutes >= 30]

    gw['is_defender']   = (gw.position == 'DEF').astype(int)
    gw['is_midfielder'] = (gw.position == 'MID').astype(int)
    gw['is_forward']    = (gw.position == 'FWD').astype(int)

    X_pred = imp.transform(gw[FEATURES])
    gw['predicted_card_prob'] = rf.predict_proba(X_pred)[:, 1].round(2)

    out_cols = ['name', 'team', 'position', 'opponent_name', 'was_home',
                'fixture_difficulty', 'predicted_card_prob']
    out_path = ROOT / f"models/GW{pred_gw}_Predicted_cards.csv"
    (gw.sort_values('predicted_card_prob', ascending=False)[out_cols]
       .to_csv(out_path, index=False))
    print("✅ Saved", out_path.name)

if __name__ == "__main__":
    main()
