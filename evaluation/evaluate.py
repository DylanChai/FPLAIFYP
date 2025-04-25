"""
Unified back‑test: rolling‑window evaluation for any target/model combo.
Example:
    python evaluate.py --models rf xgb --targets goals assists
"""

import argparse, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost            import XGBRegressor

MODELS = {
    "rf":  lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "xgb": lambda: XGBRegressor(n_estimators=300, learning_rate=0.05,
                                max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                objective="reg:squarederror", random_state=42)
}

DEFAULT_FEATS = ["minutes", "xP", "threat", "creativity"]

def rolling_backtest(df, cutoff, window, model_fn, target):
    """train on GW<=cutoff, test on (cutoff+1 … cutoff+window)"""
    train = df[df["GW"] <= cutoff]
    test  = df[(df["GW"] > cutoff) & (df["GW"] <= cutoff+window)]
    if train.empty or test.empty:
        return None
    Xtr, ytr = train[DEFAULT_FEATS], train[target]
    Xte, yte = test[DEFAULT_FEATS],  test[target]
    mdl = model_fn(); mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xte)
    return {
        "mae": mean_absolute_error(yte, pred),
        "rmse": np.sqrt(mean_squared_error(yte, pred)),
        "r2":  r2_score(yte, pred),
        "n_train": len(train), "n_test": len(test)
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models",  nargs="+", default=["rf"])
    p.add_argument("--targets", nargs="+", default=["goals_scored"])
    p.add_argument("--cutoff_gws", nargs="+", type=int, default=[10,15,20])
    p.add_argument("--windows",     nargs="+", type=int, default=[1,2])
    args = p.parse_args()

    df = pd.read_csv(Path(__file__).parents[1] /
                 "data/processed/merged_gw_cleaned.csv",
                 engine="python", on_bad_lines="skip")
    out = []
    for tgt in args.targets:
        for m in args.models:
            for c in args.cutoff_gws:
                for w in args.windows:
                    res = rolling_backtest(df, c, w, MODELS[m], tgt)
                    if res:
                        out.append(dict(model=m, target=tgt,
                                        cutoff_gw=c, window=w, **res))
    res_df = pd.DataFrame(out)
    res_df.to_csv("evaluation_results.csv", index=False, float_format="%.4g")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
