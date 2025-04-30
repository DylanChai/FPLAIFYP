# tools/fetch_shirts.py
from pathlib import Path          # built-in
import requests                   # pip install requests
import pandas as pd               # comes with Anaconda
from tqdm import tqdm      # ← this line changed

ROOT = Path(__file__).resolve().parents[1]
out  = ROOT / "models/static/shirts"
out.mkdir(parents=True, exist_ok=True)

teams = pd.read_csv(ROOT / "data/processed/teams.csv")

for _, row in tqdm(teams.iterrows(), total=len(teams)):
    url  = (f"https://fantasy.premierleague.com/dist/img/shirts/standard/"
            f"shirt_{row['short_name']}_1-66.webp")
    dest = out / f"{row['name']}.webp"
    if not dest.exists():
        r = requests.get(url, timeout=10)
        if r.ok:
            dest.write_bytes(r.content)
print("Shirt icons saved in", out)
