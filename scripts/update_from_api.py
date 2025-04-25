# ───  src/update_from_api.py  ─────────────────────────────────────────
"""
Add missing Gameweeks (22‑33) to data/processed/merged_gw_cleaned.csv
using the official Fantasy Premier League API.

Run:
    python src/update_from_api.py
"""

import time, requests, pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
PROC_CSV = ROOT / "data" / "processed" / "merged_gw_cleaned.csv"
RAW_DIR  = ROOT / "external" / "Fantasy-Premier-League" / "data" / "2024-25" / "gws"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #
def fetch_json(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def gw_stats(gw):
    """Return DF of per‑player stats for gw (id, minutes, goals, etc.)."""
    j = fetch_json(f"https://fantasy.premierleague.com/api/event/{gw}/live/")
    el = pd.json_normalize(j["elements"])
    el["GW"] = gw
    return el

def bootstrap_meta():
    j = fetch_json("https://fantasy.premierleague.com/api/bootstrap-static/")
    players = pd.json_normalize(j["elements"])
    teams   = pd.json_normalize(j["teams"])[["id", "name"]]
    return players, teams

def tidy_columns(df_stats, meta_players, meta_teams):
    # Merge names, team names, positions
    out = df_stats.merge(meta_players[["id","web_name","team","element_type"]],
                         left_on="id", right_on="id", how="left")
    out = out.merge(meta_teams, left_on="team", right_on="id", suffixes=("", "_drop"))
    out.drop(columns=["team_drop"], inplace=True)

    # Rename to Vaastav schema (subset we actually model on)
    rename = {
        "web_name": "name",
        "name":     "team",
        "element_type": "position_id",
        "stats.minutes": "minutes",
        "stats.goals_scored": "goals_scored",
        "stats.assists": "assists",
        "stats.clean_sheets": "clean_sheets",
        "stats.yellow_cards": "yellow_cards",
        "stats.red_cards": "red_cards",
        "stats.saves": "saves",
        "stats.total_points": "total_points",
        "stats.bonus": "bonus",
        "stats.bps": "bps",
        "stats.ict_index": "ict_index",
        "stats.creativity": "creativity",
        "stats.influence": "influence",
        "stats.threat": "threat",
        "stats.expected_goals": "expected_goals",
        "stats.expected_assists": "expected_assists",
        "stats.expected_goal_involvements": "expected_goal_involvements",
        "stats.expected_goals_conceded": "expected_goals_conceded",
    }
    out = out.rename(columns=rename)

    # Some columns Vaastav provides but API doesn’t → fill with NaN / 0
    fillers = ["xP","transfers_in","transfers_out","transfers_balance",
               "value","own_goals","penalties_saved","penalties_missed",
               "selected","fixture","goals_conceded","was_home",
               "team_a_score","team_h_score","opponent_team","kickoff_time",
               "round"]
    for col in fillers:
        if col not in out.columns:
            out[col] = pd.NA

    # Map position_id to MID/DEF/etc. for compatibility
    pos_map = {1:"GK",2:"DEF",3:"MID",4:"FWD"}
    out["position"] = out["position_id"].map(pos_map)

    # Order columns roughly like Vaastav (unimportant, but tidy)
    core = ["name","position","team","GW","minutes","goals_scored","assists",
            "clean_sheets","yellow_cards","red_cards","saves","total_points",
            "ict_index","creativity","influence","threat"]
    extra = [c for c in out.columns if c not in core]
    return out[core+extra]

# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    if not PROC_CSV.exists():
        raise SystemExit(f"❌ {PROC_CSV} not found – run make_dataset.py first.")

    df_base = pd.read_csv(PROC_CSV, engine="python", on_bad_lines="skip")
    present_gws = set(df_base["GW"].unique())
    target_gws  = list(range(22, 34))                       # up to GW 33
    missing_gws = [gw for gw in target_gws if gw not in present_gws]

    if not missing_gws:
        print("Everything up‑to‑date – no missing GWs.")
        raise SystemExit

    players_meta, teams_meta = bootstrap_meta()

    new_rows = []
    for gw in missing_gws:
        try:
            print(f"📡  Fetching GW {gw} …")
            df_stats = gw_stats(gw)
            df_tidy  = tidy_columns(df_stats, players_meta, teams_meta)
            new_rows.append(df_tidy)
            # optional raw snapshot:
            df_tidy.to_csv(RAW_DIR/f"gw{gw}.csv", index=False)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️  GW {gw} skipped ({e})")

    if not new_rows:
        print("No rows downloaded – exiting.")
        raise SystemExit

    df_all = pd.concat([df_base] + new_rows, ignore_index=True)
    df_all.to_csv(PROC_CSV, index=False)
    print(f"✅  Added {len(pd.concat(new_rows)):,} rows. Highest GW now: "
          f"{df_all['GW'].max()}")
# ────────────────────────────────────────────────────────────────────
