# tools/fetch_badges.py
from pathlib import Path
import pandas as pd, requests, tqdm

ROOT   = Path(__file__).resolve().parents[1]
outdir = ROOT / "models" / "static" / "badges"
outdir.mkdir(parents=True, exist_ok=True)

teams = pd.read_csv(ROOT / "data/processed/teams.csv")
for _, row in tqdm.tqdm(teams.iterrows(), total=len(teams)):
    url  = f"https://resources.premierleague.com/premierleague/badges/50/t{row['code']}.png"
    dest = outdir / f"{row['name']}.png"
    if not dest.exists():
        r = requests.get(url, timeout=10)
        if r.ok: dest.write_bytes(r.content)
print("Badges saved in", outdir)
