"""
Fetch current season fixtures and save as data/processed/fixtures.csv
"""

import requests, pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "fixtures.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

url = "https://fantasy.premierleague.com/api/fixtures/"
r = requests.get(url, timeout=20)
fixtures = pd.DataFrame(r.json())

# Parse and keep only relevant fields
fixtures = fixtures[["event", "kickoff_time", "team_h", "team_a"]]
fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"])
fixtures = fixtures.dropna(subset=["kickoff_time"])

fixtures.to_csv(OUT, index=False)
print(f"✅  Saved {len(fixtures)} fixtures to {OUT}")
print(f"📅  GW range: {fixtures['event'].min()} – {fixtures['event'].max()}")
