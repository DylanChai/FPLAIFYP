"""
Scrape EPL match odds from the‑odds‑api.com and tag each row with the FPL Gameweek.

Requires:
    * the‑odds‑api key (set in API_KEY)
    * data/processed/fixtures.csv  (produced by make_dataset.py)

Save path:
    data/processed/epl_betting_odds.csv
"""

import requests, pandas as pd, os, sys
from pathlib import Path

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
API_KEY = "c94869c70bbfd926f3fa4ffa9bb0aded"  
SPORT   = "soccer_epl"
REGION  = "uk"
MARKET  = "h2h"
ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

ODDS_CSV = PROC / "epl_betting_odds.csv"
FIXT_CSV = PROC / "fixtures.csv"

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def odds_to_prob(decimal):
    return 1.0 / decimal if decimal else 0.0

def normalise(p_home, p_draw, p_away):
    total = p_home + p_draw + p_away
    return (p_home/total, p_draw/total, p_away/total) if total else (0,0,0)

def fetch_odds():
    url = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
           f"?apiKey={API_KEY}&regions={REGION}&markets={MARKET}&oddsFormat=decimal")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def tag_with_gw(df):
    if not FIXT_CSV.exists():
        print(f"❌  {FIXT_CSV} not found. Run make_dataset.py first.")
        sys.exit(1)

    fix = pd.read_csv(FIXT_CSV, parse_dates=["kickoff_time"])
    fix["kickoff_time"] = fix["kickoff_time"].dt.tz_localize(None).dt.round("min")
    df["kickoff"] = pd.to_datetime(df["kickoff"]).dt.round("min")

    # ✅ Print last 5 times from each
    print("\n📆 Fixture times (rounded):")
    print(fix["kickoff_time"].sort_values().tail(5))

    print("\n🕒 Odds times (rounded):")
    print(df["kickoff"].sort_values().tail(5))

    ts2gw = dict(zip(fix["kickoff_time"], fix["event"]))
    df["GW"] = df["kickoff"].apply(lambda x: ts2gw.get(x))

    missing = df["GW"].isna().sum()
    if missing:
        print(f"⚠️  {missing} rows could not be matched to a GW and will be dropped")
        df = df.dropna(subset=["GW"])

    df["GW"] = df["GW"].astype(int)
    return df


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("📡  Fetching odds …")
    odds_json = fetch_odds()

    rows = []
    for match in odds_json:
        if not match.get("bookmakers"):
            continue
        site = match["bookmakers"][0]
        outcomes = site["markets"][0]["outcomes"]
        odds = {o["name"]: o["price"] for o in outcomes}

        home, away = match["home_team"], match["away_team"]
        p_home = odds_to_prob(odds.get(home))
        p_draw = odds_to_prob(odds.get("Draw"))
        p_away = odds_to_prob(odds.get(away))
        n_home, n_draw, n_away = normalise(p_home, p_draw, p_away)

        rows.append({
            "home_team": home,
            "away_team": away,
            "kickoff":   match["commence_time"],
            "prob_home": round(n_home, 4),
            "prob_draw": round(n_draw, 4),
            "prob_away": round(n_away, 4),
            "bookmaker": site["title"]
        })

    df = pd.DataFrame(rows)
    df = tag_with_gw(df)
    df.to_csv(ODDS_CSV, index=False)
    print(f"✅  Saved odds to {ODDS_CSV} ({len(df)} rows, GWs {df['GW'].unique().tolist()})")
