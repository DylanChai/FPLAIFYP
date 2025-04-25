"""
Optimal FPL squad builder
─────────────────────────
• 15 players  = 2 GK 5 DEF 5 MID 3 FWD
• exactly 11 starters
• total price ≤ £100 m
• NO more than **3 players per real club**
• maximises expected GW points from your four prediction CSVs
Outputs:
    GWxx_OptimalSquad.csv   – full 15 (XI column)
    GWxx_OptimalXI.csv      – starting XI only
"""

import pandas as pd, numpy as np
from pathlib import Path
import pulp, sys

ROOT = Path(__file__).resolve().parents[1]

# ───────── helper ───────────────────────────────────────────────
def newest(pattern, folder=ROOT / "models"):
    files = list(folder.glob(pattern))
    if not files:
        sys.exit(f"[Opt] No files match {pattern}")
    files.sort(key=lambda p: int(p.stem.split("_")[0][2:]), reverse=True)
    return files[0]

# ───────── load predictions ─────────────────────────────────────
goals   = pd.read_csv(newest("GW*_Predicted_goals_with_fixtures.csv"))
assists = pd.read_csv(newest("GW*_Predicted_assists.csv"))
cs      = pd.read_csv(newest("GW*_Predicted_[cC]lean_[sS]heets.csv"))
cards   = pd.read_csv(newest("GW*_Predicted_cards.csv"))
gw      = int(newest("GW*_Predicted_goals_with_fixtures.csv").stem.split('_')[0][2:])

# ───────── load price / position master ─────────────────────────
master = pd.read_csv(ROOT / "data/processed/merged_gw_cleaned.csv",
                     on_bad_lines="skip")
price_pos = (master.sort_values("GW")
                    .groupby("name")[["position", "value", "team"]]
                    .last()
                    .reset_index())
price_pos["price"] = price_pos["value"] / 10
price_pos.drop(columns="value", inplace=True)

# ───────── merge & harmonise columns ────────────────────────────
def keep(left, right, col):  # one-column safe merge
    return left.merge(right[["name", col]], on="name", how="left")

df = (goals.rename(columns={"predicted_goals": "GoalsProb"})
          .pipe(keep, assists.rename(columns={"predicted_assists": "AstProb"}), "AstProb")
          .pipe(keep, cs, cs.columns[-1])     # last col == cs prob
          .pipe(keep, cards.rename(columns={"predicted_card_prob": "CardProb"}), "CardProb")
          .merge(price_pos, on="name", how="left"))

legacy = {"predicted_cs_prob": "CSProb", "clean_sheet_probability": "CSProb"}
df.rename(columns={k: v for k, v in legacy.items() if k in df.columns}, inplace=True)

for col, default in {"GoalsProb": 0, "AstProb": 0, "CSProb": 0, "CardProb": 0}.items():
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(0)

# minutes fallback
if "roll3_minutes" in df.columns:
    df["Minutes"] = df["roll3_minutes"].fillna(60)
elif "minutes" in df.columns:
    df["Minutes"] = df["minutes"].fillna(60)
else:
    df["Minutes"] = 60

# position guarantee
if "position" not in df.columns:
    for alt in ("position_x", "position_y"):
        if alt in df.columns:
            df["position"] = df[alt]
            break
    else:
        df["position"] = "MID"
df["position"] = df["position"].fillna("MID")

# ── team guarantee (needed for max-3-per-club) ─────────────────
if 'team' not in df.columns:
    for alt in ('team_x', 'team_y'):
        if alt in df.columns:
            df['team'] = df[alt]
            break
if 'team' not in df.columns:
    df['team'] = 'Unknown'
df['team'] = df['team'].fillna('Unknown')


df.dropna(subset=["price"], inplace=True)  # must have price £

# ───────── expected points model ────────────────────────────────
def g_pts(pos): return 5 if pos == "MID" else 4
def cs_pts(pos): return 4 if pos in ("GK", "DEF") else 0

df["ExpPts"] = (
    df["GoalsProb"] * df["position"].map(g_pts)
    + df["AstProb"] * 3
    + df["CSProb"]  * df["position"].map(cs_pts)
    + np.where(df["Minutes"] >= 60, 2, 0)
    + df["CardProb"] * -1
)

# ───────── optimisation (MILP) ──────────────────────────────────
m = pulp.LpProblem("FPL_Optimiser", pulp.LpMaximize)
x = pulp.LpVariable.dicts("squad", df.index, 0, 1, cat="Binary")
y = pulp.LpVariable.dicts("xi",    df.index, 0, 1, cat="Binary")

m += pulp.lpSum(df.loc[i, "ExpPts"] * y[i] for i in df.index)

# squad size, budget
m += pulp.lpSum(x[i] for i in df.index) == 15
m += pulp.lpSum(df.loc[i, "price"] * x[i] for i in df.index) <= 100

# position quotas
for p, q in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
    m += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == p) == q

# max 3 per real club
for club in df["team"].unique():
    m += pulp.lpSum(x[i] for i in df.index if df.loc[i, "team"] == club) <= 3

# XI rules
m += pulp.lpSum(y[i] for i in df.index) == 11
m += pulp.lpSum(y[i] for i in df.index if df.loc[i, "position"] == "GK") == 1
m += pulp.lpSum(y[i] for i in df.index if df.loc[i, "position"] == "DEF") >= 3
m += pulp.lpSum(y[i] for i in df.index if df.loc[i, "position"] == "MID") >= 2
m += pulp.lpSum(y[i] for i in df.index if df.loc[i, "position"] == "FWD") >= 1
for i in df.index:
    m += y[i] <= x[i]

m.solve(pulp.PULP_CBC_CMD(msg=False))
print("Status:", pulp.LpStatus[m.status])

picked = df[["name", "team", "position", "price", "ExpPts"]].copy()
picked["Squad"] = [int(x[i].value()) for i in df.index]
picked["XI"]    = [int(y[i].value()) for i in df.index]
squad = picked[picked.Squad == 1].sort_values("XI", ascending=False)

out = ROOT / "models"
squad.to_csv(out / f"GW{gw}_OptimalSquad.csv", index=False)
squad[squad.XI == 1].to_csv(out / f"GW{gw}_OptimalXI.csv", index=False)
print(f"Saved: GW{gw}_OptimalSquad.csv  &  GW{gw}_OptimalXI.csv")
