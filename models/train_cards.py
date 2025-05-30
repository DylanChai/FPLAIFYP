"""
FPL Card-Probability Model  (yellow OR red in next GW)

Creates models/GW<xx>_Predicted_cards.csv   with columns
  name, team, position, opponent_name, was_home,
  fixture_difficulty, predicted_card_prob
"""

from pathlib import Path
import pandas as pd, numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, brier_score_loss

# helpers from goals script
from train_goalsTeamStrength import enhance_with_team_data, prepare_gw_data

ROOT   = Path(__file__).resolve().parents[1]
DATA   = pd.read_csv(ROOT / "data/processed/merged_gw_cleaned.csv",
                     on_bad_lines="skip")
TEAMS  = pd.read_csv(ROOT / "data/processed/teams.csv")
FIX    = pd.read_csv(ROOT / "data/processed/fixtures.csv",
                     parse_dates=["kickoff_time"])

now          = pd.Timestamp.now(tz="UTC")   # Current time
last_played  = int(FIX[FIX.kickoff_time <= now]["event"].max()) # Latest completed GW
PRED_GW      = last_played + 1  # GW we try forecast for

# Training on 1-32, minus outfield players
TR = DATA[(DATA.GW.between(1, min(last_played, 32))) &
          (DATA.position != "GK")].copy()

for col in ["yellow_cards", "red_cards", "minutes"]:
    TR[f"roll3_{col}"] = (TR.groupby("name")[col]
                            .transform(lambda x: x.rolling(3, 1).mean()))

TR = enhance_with_team_data(TR, TEAMS)

TR["is_defender"]   = (TR.position == "DEF").astype(int)
TR["is_midfielder"] = (TR.position == "MID").astype(int)
TR["is_forward"]    = (TR.position == "FWD").astype(int)

FEATS = [
    "roll3_minutes", "roll3_yellow_cards", "roll3_red_cards",
    "team_attacking", "opp_defending", "fixture_difficulty", "was_home",
    "is_defender", "is_midfielder", "is_forward"
]
TR[FEATS] = TR[FEATS].apply(pd.to_numeric, errors="coerce")
TR["got_card"] = ((TR.yellow_cards + TR.red_cards) > 0).astype(int)

imp = SimpleImputer(strategy="mean")
X   = imp.fit_transform(TR[FEATS])
y   = TR["got_card"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf_base = RandomForestClassifier(
    n_estimators=400, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1
)

rf_base.fit(X_tr, y_tr)

# ✅ Calculate and print Brier score on test set
y_proba = rf_base.predict_proba(X_te)[:, 1]
brier = brier_score_loss(y_te, y_proba)
print(f"Brier score: {brier:.5f}")

# ✅ Save feature importance
importance_df = pd.DataFrame({
    'feature': FEATS,
    'importance': rf_base.feature_importances_
}).sort_values(by='importance', ascending=False)

importance_path = ROOT / "models" / "cards_feature_importance.csv"
importance_df.to_csv(importance_path, index=False)
print(f"✅ Feature importance saved to {importance_path.name}")

players  = TR[["name", "team", "position"]].drop_duplicates()
num_cols = TR.select_dtypes(np.number).columns
roll_avg = TR.groupby(["name", "team"])[num_cols].mean().reset_index()

GW = prepare_gw_data(players, roll_avg, TEAMS, FIX, PRED_GW)
# Drop players whos opponent is unknown this stops players who have no fixtures in blank GW's Appearing and dropping GK
GW = GW[(GW.opponent_name != "Unknown") & (GW.position != "GK")]

if "roll3_minutes" not in GW.columns:
    GW["roll3_minutes"] = GW["minutes"]
    # Filter out players unlikely to play
GW = GW[GW.roll3_minutes >= 30]

GW["is_defender"]   = (GW.position == "DEF").astype(int)
GW["is_midfielder"] = (GW.position == "MID").astype(int)
GW["is_forward"]    = (GW.position == "FWD").astype(int)

X_pred = imp.transform(GW[FEATS])
raw = rf_base.predict_proba(X_pred)[:, 1]

# ---- scale 99th-percentile to 0.35, then clip ----------------------
p99        = np.percentile(raw, 99)
scaled     = (raw / p99) * 0.35
scaled     = np.clip(scaled, 0, 0.35)
GW["predicted_card_prob"] = scaled.round(2)

# ---------- save ----------------------------------------------------
OUT_COLS = ["name", "team", "position", "opponent_name", "was_home",
            "fixture_difficulty", "predicted_card_prob"]
out_path = ROOT / f"models/GW{PRED_GW}_Predicted_cards.csv"

(GW.sort_values("predicted_card_prob", ascending=False)[OUT_COLS]
   .to_csv(out_path, index=False))
print("✅ Saved", out_path.name)
