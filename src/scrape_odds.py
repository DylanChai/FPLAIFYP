import requests
import pandas as pd
import os

# ------------------------------
# CONFIGURATION
# ------------------------------
API_KEY = "c94869c70bbfd926f3fa4ffa9bb0aded"  # <- Replace this with your API key from https://the-odds-api.com
SPORT = "soccer_epl"
REGION = "uk"
MARKET = "h2h"  # head-to-head odds
ODDS_CSV_PATH = "epl_betting_odds.csv"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def odds_to_prob(odds):
    return 1 / odds if odds else 0

def normalize_probs(home, draw, away):
    total = home + draw + away
    if total == 0:
        return 0, 0, 0
    return home / total, draw / total, away / total

# ------------------------------
# MAIN SCRAPING LOGIC
# ------------------------------
def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGION}&markets={MARKET}&oddsFormat=decimal"
    response = requests.get(url)

    if response.status_code != 200:
        print("❌ Failed to fetch odds:", response.status_code, response.text)
        return None

    odds_data = response.json()
    records = []

    for match in odds_data:
        home = match["home_team"]
        away = match["away_team"]
        match_time = match["commence_time"]

        # Skip matches with no odds yet
        if not match.get("bookmakers"):
            continue

        site = match["bookmakers"][0]

        try:
            outcomes = site["markets"][0]["outcomes"]
            odds_dict = {o["name"]: o["price"] for o in outcomes}

            home_odds = odds_dict.get(home)
            draw_odds = odds_dict.get("Draw")
            away_odds = odds_dict.get(away)

            p_home = odds_to_prob(home_odds)
            p_draw = odds_to_prob(draw_odds)
            p_away = odds_to_prob(away_odds)

            norm_home, norm_draw, norm_away = normalize_probs(p_home, p_draw, p_away)

            records.append({
                "home_team": home,
                "away_team": away,
                "kickoff": match_time,
                "prob_home_win": round(norm_home, 4),
                "prob_draw": round(norm_draw, 4),
                "prob_away_win": round(norm_away, 4),
                "bookmaker": site["title"]
            })

        except Exception as e:
            print(f"⚠️ Skipping {home} vs {away} due to error: {e}")
            continue

    return pd.DataFrame(records)

# ------------------------------
# ENTRY POINT
# ------------------------------
def main():
    df = fetch_odds()
    if df is not None and not df.empty:
        df.to_csv(ODDS_CSV_PATH, index=False)
        print(f"\n✅ Betting odds saved to: {ODDS_CSV_PATH}")
        print(df.head())
    else:
        print("⚠️ No odds data available to save.")

if __name__ == "__main__":
    main()
