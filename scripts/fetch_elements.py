import pathlib
import requests
import pandas as pd

BASE = pathlib.Path(__file__).resolve()

master_csv = BASE.parent.parent / "data/processed/elements.csv"

if not master_csv.exists():
    print("Downloading fresh elements.csv...")
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        timeout=10
    ).json()
    elements = pd.json_normalize(bootstrap["elements"])
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    elements.to_csv(master_csv, index=False)

print(f"elements.csv saved to {master_csv}")
