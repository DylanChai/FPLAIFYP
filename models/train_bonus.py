"""
FPL Bonus-Point Prediction Model  (0-3 bonus points next GW)

Creates  GW<nn>_Predicted_bonus.csv  with:
  name, team, position, opponent_name, was_home,
  fixture_difficulty, predicted_bonus

"""
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Re-use helpers from goals script
from train_goalsTeamStrength import enhance_with_team_data, prepare_gw_data

# ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ---------- load master data -------------------------------------------
DATA  = pd.read_csv(ROOT / "data/processed/merged_gw_cleaned.csv",
                    on_bad_lines="skip")
TEAMS = pd.read_csv(ROOT / "data/processed/teams.csv")
FIX   = pd.read_csv(ROOT / "data/processed/fixtures.csv",
                    parse_dates=["kickoff_time"])

# ---------- find current / next GW -------------------------------------
now_utc = pd.Timestamp.now(tz="UTC")
LAST    = int(FIX[FIX.kickoff_time <= now_utc]["event"].max())
PRED_GW = LAST + 1

# ---------- training slice (GW 1-32) -----------------------------------
tr = DATA[DATA.GW.between(1, min(LAST, 32))].copy()

tr.replace({"True": 1, "False": 0}, inplace=True)

ROLL_COLS = ['minutes', 'goals_scored', 'assists', 'clean_sheets',
             'saves', 'yellow_cards', 'red_cards',
             'threat', 'creativity', 'influence', 'ict_index']

for col in ROLL_COLS:
    tr[col] = pd.to_numeric(tr[col], errors='coerce')
    tr[f'roll3_{col}'] = (tr.groupby('name')[col]
                            .transform(lambda s: s.rolling(3, 1).mean()))

tr = enhance_with_team_data(tr, TEAMS)

tr['is_forward']    = (tr.position == 'FWD').astype(int)
tr['is_midfielder'] = (tr.position == 'MID').astype(int)
tr['is_defender']   = (tr.position == 'DEF').astype(int)
tr['is_goalkeeper'] = (tr.position == 'GK').astype(int)

FEATS = [c for c in tr.columns if c.startswith('roll3_')] + [
        'team_attacking', 'opp_defending', 'fixture_difficulty',
        'is_forward', 'is_midfielder', 'is_defender', 'is_goalkeeper'
]
TARGET = 'bonus'

tr[FEATS]  = tr[FEATS].apply(pd.to_numeric, errors='coerce')
tr[TARGET] = pd.to_numeric(tr[TARGET],  errors='coerce')
tr.dropna(subset=[TARGET], inplace=True)

# ---------- fit Random-Forest regressor ---------------------------------
imp = SimpleImputer(strategy='mean')
X   = imp.fit_transform(tr[FEATS])
y   = tr[TARGET]

X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5,
        max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

print("MAE (val):", round(mean_absolute_error(y_te, rf.predict(X_te)), 4))

# --- feature importance -------------------------------------------------
fi = (pd.DataFrame({'feature': FEATS,
                    'importance': rf.feature_importances_})
        .sort_values('importance', ascending=False))
print("\nTop feature importance:")
print(fi.head(15).to_string(index=False))

# ---------- prepare upcoming GW frame ----------------------------------
players = tr[['name', 'team', 'position']].drop_duplicates()
numcols = tr.select_dtypes(np.number).columns
rollavg = tr.groupby(['name', 'team'])[numcols].mean().reset_index()

gw = prepare_gw_data(players, rollavg, TEAMS, FIX, PRED_GW)
gw = gw[gw.opponent_name != 'Unknown']      # drop blank-GW players
gw = gw[gw.roll3_minutes >= 30]             # discard cold players

for flag, pos in [('is_forward', 'FWD'),
                  ('is_midfielder', 'MID'),
                  ('is_defender', 'DEF'),
                  ('is_goalkeeper', 'GK')]:
    gw[flag] = (gw.position == pos).astype(int)

X_pred = imp.transform(gw[FEATS])
gw['predicted_bonus'] = rf.predict(X_pred).round(2)

OUT_COLS = ['name', 'team', 'position', 'opponent_name', 'was_home',
            'fixture_difficulty', 'predicted_bonus']
out_path = ROOT / f"models/GW{PRED_GW}_Predicted_bonus.csv"
gw.sort_values('predicted_bonus', ascending=False)[OUT_COLS]\
  .to_csv(out_path, index=False)

print("✅  Saved", out_path.name)
