import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

def main():
    # ─── configure paths ─────────────────────────────────────────────────────────
    BASE          = Path(__file__).resolve().parents[1]
    DATA_PATH     = BASE / "data" / "processed" / "merged_gw_cleaned.csv"
    TEAMS_PATH    = BASE / "data" / "processed" / "teams.csv"
    FIXTURES_PATH = BASE / "data" / "processed" / "fixtures.csv"

    # ─── load data ────────────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        print(f"❌ Data file not found: {DATA_PATH}")
        return
    data = pd.read_csv(DATA_PATH, on_bad_lines="skip")
    print(f"✅ Loaded data: {data.shape}")

    teams_df    = pd.read_csv(TEAMS_PATH)
    fixtures_df = pd.read_csv(FIXTURES_PATH, parse_dates=["kickoff_time"])
    # ensure UTC on kickoff_time
    if fixtures_df["kickoff_time"].dt.tz is None:
        fixtures_df["kickoff_time"] = fixtures_df["kickoff_time"].dt.tz_localize("UTC")

    # ─── prepare training slice ─────────────────────────────────────────────────
    # use GWs 1–30 for training
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 30)].copy()
    if train_data.empty:
        print("❌ No training data for GW1–30")
        return
    print(f"✅ Training shape: {train_data.shape}")

    # ─── features & target ───────────────────────────────────────────────────────
    features = [
        "minutes",
        "xP",
        "threat",
        "bps",
        "transfers_in",
        "expected_goals",
    ]
    target = "goals_scored"

    # drop rows where target is missing
    train_data[target] = pd.to_numeric(train_data[target], errors="coerce")
    train_data = train_data.dropna(subset=[target])

    # coerce all features to numeric
    for f in features:
        train_data[f] = pd.to_numeric(train_data[f], errors="coerce")

    # ─── imputation ───────────────────────────────────────────────────────────────
    imp = SimpleImputer(strategy="mean")
    X_all = imp.fit_transform(train_data[features].fillna(0))
    y_all = train_data[target]

    # ─── train/test split + model ────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    # evaluate
    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    print(f"📉 MAE on test set: {mae:.5f}")

    # ─── predict next GW ─────────────────────────────────────────────────────────
    # get GW31 (or whatever next GW is)
    latest_gw = int(data["GW"].max()) + 1

    # prepare “future” dataset: one row per player
    players = data[["name","team"]].drop_duplicates()
    num_cols = train_data.select_dtypes(include="number").columns.tolist()
    avg_stats = train_data.groupby(["name","team"])[features].mean().reset_index()

    future = players.merge(avg_stats, on=["name","team"], how="left")
    future["GW"] = latest_gw

    # impute missing
    X_fut = imp.transform(future[features].fillna(0))
    raw_preds = model.predict(X_fut)

    # floor at .001
    future["predicted_goals"] = np.maximum(raw_preds, 0.001)

    # sort & save
    out = BASE / "models" / f"GW{latest_gw}_Predicted_goals.csv"
    future.sort_values("predicted_goals", ascending=False)\
          .to_csv(out, index=False)
    print(f"✅ Saved predictions to {out.name}")

if __name__ == "__main__":
    main()
