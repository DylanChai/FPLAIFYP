# author: Dylan
# source data from: https://github.com/vaastav/Fantasy-Premier-League
# purpose: Predict [cards | clean sheets | assits | goals] for Fantasy Premier League using ML


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("Starting improved assists prediction model...")
    
    # 1. Load the merged gameweek file for the 2024-25 season
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "external", "Fantasy-Premier-League", "data", "2024-25", "gws", "merged_gw_cleaned.csv")

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        print("Please run the CSV cleaner script first to create the cleaned CSV file.")
        return

    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"Successfully loaded data with shape: {data.shape}")
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return

    # 2. Use gameweeks 1-30 as training data (UPDATED)
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 30)]
    if train_data.empty:
        print("No training data found for GW 1-30 in merged_gw_cleaned.csv.")
        return
    print(f"Training data shape (GW 1-30): {train_data.shape}")

    # 3. Add feature: recent form (last 5 gameweeks)
    print("Adding recent form feature...")
    # Sort by player and gameweek
    train_data = train_data.sort_values(["name", "GW"])
    
    # Calculate rolling assists (form)
    train_data["recent_assists_form"] = train_data.groupby("name")["assists"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate games played in the season so far
    train_data["games_played"] = train_data.groupby("name").cumcount() + 1
    
    # Calculate season assist rate 
    train_data["season_assist_rate"] = train_data.groupby("name")["assists"].transform(
        lambda x: x.expanding().sum()
    ) / train_data["games_played"]
    
    # 4. Select features for assist prediction (enhanced)
    assists_features = [
        "minutes",              # Playing time
        "creativity",           # Key FPL metric for assists
        "recent_assists_form",  # Recent form (new feature)
        "season_assist_rate",   # Assist rate so far (new feature)
        "bps",                  # Bonus points (related to key passes, etc.)
        "influence",            # General match influence
        "total_points"          # Overall performance level
    ]
    
    # Add expected metrics if available
    optional_features = ["expected_assists", "expected_goal_involvements", "xP"]
    for feature in optional_features:
        if feature in train_data.columns:
            assists_features.append(feature)
    
    # Check if all features exist in the dataset
    missing_features = [f for f in assists_features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        assists_features = [f for f in assists_features if f in train_data.columns]
    
    print(f"Using these features: {assists_features}")
    
    target_assists = "assists"

    # 5. Position encoding - keeping track for later scaling
    if "position" in train_data.columns:
        train_data = pd.get_dummies(train_data, columns=["position"])
        position_cols = [col for col in train_data.columns if col.startswith("position_")]
        assists_features.extend(position_cols)
        # Save position distribution for later scaling
        position_assist_rates = {}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            pos_col = f"position_{pos}"
            if pos_col in train_data.columns:
                # Calculate average assists per game for this position
                position_data = train_data[train_data[pos_col] == 1]
                if not position_data.empty:
                    position_assist_rates[pos] = position_data["assists"].mean()
                    print(f"Average assists for {pos}: {position_assist_rates[pos]:.4f} per game")
    else:
        print("Position column not found. Continuing without position encoding.")

    # 6. Drop rows with missing values
    train_data = train_data.dropna(subset=assists_features + [target_assists])
    for col in assists_features + [target_assists]:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 7. Split data and train model
    X_assists = train_data[assists_features]
    y_assists = train_data[target_assists]
    
    X_train, X_test, y_train, y_test = train_test_split(X_assists, y_assists, test_size=0.2, random_state=42)
    
    model_assists = RandomForestRegressor(n_estimators=100, random_state=42)
    model_assists.fit(X_train, y_train)
    
    # 8. Evaluate model
    predictions_test = model_assists.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")
    
    # 9. Prepare GW 31 data
    print("Preparing GW 31 predictions...")
    
    # Get most recent player data for form features
    latest_data = train_data.sort_values("GW", ascending=False).groupby("name").first().reset_index()
    latest_features = latest_data[["name", "team", "recent_assists_form", "season_assist_rate"]]
    
    # Get player average stats
    player_avgs = train_data.groupby(["name", "team"])[
        [col for col in assists_features if col not in ["recent_assists_form", "season_assist_rate"]]
    ].mean().reset_index()
    
    # Merge latest form with average stats
    gw31_data = pd.merge(player_avgs, latest_features, on=["name", "team"])
    gw31_data["GW"] = 31
    
    # 10. Make predictions
    X_future = gw31_data[assists_features]
    raw_predictions = model_assists.predict(X_future)
    
    # 11. Post-process predictions for realism
    
    # Store team-level info for better scaling
    team_creative_strength = defaultdict(float)
    for team in train_data["team"].unique():
        team_data = train_data[train_data["team"] == team]
        if not team_data.empty:
            team_creative_strength[team] = team_data["creativity"].mean() / train_data["creativity"].mean()
    
    # Calculate team strength adjustment factors
    print("\nTeam creative strength factors:")
    for team, factor in sorted(team_creative_strength.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{team}: {factor:.2f}x")
    
    # Calculate realistic assists predictions with better scaling
    scaled_predictions = []
    
    # Create player position lookup
    player_positions = {}
    if "position_DEF" in train_data.columns:
        for _, row in train_data.drop_duplicates(subset=["name"]).iterrows():
            pos = "Unknown"
            if row.get("position_GK", 0) == 1: pos = "GK"
            elif row.get("position_DEF", 0) == 1: pos = "DEF"
            elif row.get("position_MID", 0) == 1: pos = "MID"
            elif row.get("position_FWD", 0) == 1: pos = "FWD"
            player_positions[row["name"]] = pos
    
    # Define realistic per-game assist ranges by position
    position_max_assists = {
        "GK": 0.08,      # ~3 per season max
        "DEF": 0.15,     # ~5-6 per season max
        "MID": 0.3,      # ~11-12 per season max
        "FWD": 0.25,     # ~9-10 per season max
        "Unknown": 0.2   # Default if position unknown
    }
    
    # Find maximum raw prediction for scaling
    max_raw_prediction = max(raw_predictions)
    if max_raw_prediction == 0:
        max_raw_prediction = 1  # Prevent division by zero
    
    # Apply position-based scaling
    for i, pred in enumerate(raw_predictions):
        player_name = gw31_data.iloc[i]["name"]
        player_team = gw31_data.iloc[i]["team"]
        
        # Get player position
        player_pos = player_positions.get(player_name, "Unknown")
        
        # Get team creative strength factor
        team_factor = team_creative_strength.get(player_team, 1.0)
        
        # Get position-specific max assists
        pos_max = position_max_assists.get(player_pos, 0.2)
        
        # Scale prediction relative to maximum and apply position cap
        # Using sigmoid-like scaling to prevent excessive values
        relative_strength = pred / max_raw_prediction
        scaled_value = pos_max * (2 / (1 + np.exp(-5 * relative_strength)) - 1) * team_factor
        
        # Ensure reasonable minimum and add slight randomization
        randomization = np.random.uniform(-0.01, 0.01)
        final_value = max(0, scaled_value + randomization)
        
        scaled_predictions.append(final_value)
    
    # Add scaled predictions to dataframe
    gw31_data["predicted_assists"] = scaled_predictions
    
    # 12. Final processing and output
    # Sort by predicted assists
    result_df = gw31_data.sort_values(by="predicted_assists", ascending=False)
    
    # Check distribution of predictions
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    hist, _ = np.histogram(result_df["predicted_assists"], bins=bins)
    
    print("\nDistribution of assist predictions (per game):")
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]} players")
    
    # Save predictions
    out_path = "GW31_Predicted_assists.csv"
    result_df[["name", "team", "GW", "predicted_assists"]].to_csv(out_path, index=False)
    
    print(f"\nSaved predictions to {out_path}")
    print("Top 10 predicted assisters:")
    print(result_df[["name", "team", "predicted_assists"]].head(10))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': assists_features,
        'Importance': model_assists.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature importance:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()