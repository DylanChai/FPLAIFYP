import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("Starting clean sheets prediction model...")

    # 1. Load the merged gameweek file using the cleaned CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "external", "Fantasy-Premier-League", "data", "2024-25", "gws", "merged_gw_cleaned.csv")

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    try:
        data = pd.read_csv(data_path, engine='python', on_bad_lines='skip')
        print(f"Successfully loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 2. Updated to use gameweeks 1-30 as training data
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 30)]
    if train_data.empty:
        print("No training data found for GW 1-30.")
        return
    print(f"Training data shape (GW 1-30): {train_data.shape}")

    # 3. Convert columns to numeric
    for col in train_data.columns:
        if col not in ["name", "team", "position", "kickoff_time"]:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 4. Check if clean_sheets column exists
    if "clean_sheets" not in train_data.columns:
        print("Creating clean sheets column from goals conceded...")
        if "goals_conceded" in train_data.columns:
            train_data["clean_sheets"] = (train_data["goals_conceded"] == 0).astype(int)
        else:
            print("Error: Cannot create clean sheets column. Exiting.")
            return

    # 5. Select features for clean sheets prediction
    clean_sheet_features = [
        "minutes",
        "total_points",
        "goals_conceded"
    ]
    
    # Add more features if available
    optional_features = ["bps", "influence", "was_home", "threat"]
    for feature in optional_features:
        if feature in train_data.columns:
            clean_sheet_features.append(feature)
    
    # Check if all features exist
    missing_features = [f for f in clean_sheet_features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing: {missing_features}")
        clean_sheet_features = [f for f in clean_sheet_features if f in train_data.columns]
    
    print(f"Using these features: {clean_sheet_features}")
    
    # 6. Prepare data for modeling
    train_data = train_data.dropna(subset=clean_sheet_features + ["clean_sheets"])
    
    X = train_data[clean_sheet_features]
    y = train_data["clean_sheets"]
    
    # Check data balance
    clean_sheet_rate = y.mean()
    print(f"Clean sheet rate in training data: {clean_sheet_rate:.4f}")
    
    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 8. Train model
    print("Training clean sheets model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 9. Evaluate model
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model MAE: {mae:.4f}")
    
    # 10. Get GW 31 predictions (using most recent data for each player) - UPDATED
    print("Preparing GW 31 predictions...")
    
    # Group by player and team to get the most recent data
    latest_data = train_data.sort_values("GW", ascending=False).groupby(["name", "team"]).first().reset_index()
    
    # Select meaningful features
    gw31_data = latest_data[["name", "team"] + clean_sheet_features].copy()
    gw31_data["GW"] = 31  # Updated to GW 31
    
    # 11. Make predictions
    X_future = gw31_data[clean_sheet_features]
    raw_predictions = model.predict(X_future)
    
    # 12. Scale predictions to realistic range (5% to 60%)
    # This ensures we get a realistic distribution of clean sheet probabilities
    print("Scaling predictions to realistic range...")
    
    min_prob, max_prob = 0.05, 0.60  # Realistic range for EPL clean sheets
    
    # Define team tiers (manual adjustment based on current form)
    # UPDATED: You may want to adjust these based on current season performance
    top_teams = ["Man City", "Arsenal", "Liverpool","Nottingham Forest", "Bournemouth"]
    mid_teams = ["Newcastle", "Tottenham", "Chelsea", "Man Utd", "West Ham", 
                "Crystal Palace", "Brentford", "Fulham", "Wolves", "Brighton","Aston Villa"]
    bottom_teams = ["Everton","Ipswich", 
                   "Leicester", "Southampton"]
    
    # Add team tier info
    gw31_data["team_tier"] = 2  # Default to mid-tier
    for idx, row in gw31_data.iterrows():
        team = row["team"]
        if any(team_name in team for team_name in top_teams):
            gw31_data.at[idx, "team_tier"] = 1  # Top tier
        elif any(team_name in team for team_name in bottom_teams):
            gw31_data.at[idx, "team_tier"] = 3  # Bottom tier
    
    # Scale predictions based on team tier
    scaled_predictions = []
    for i, pred in enumerate(raw_predictions):
        tier = gw31_data.iloc[i]["team_tier"]
        if tier == 1:  # Top tier
            # Scale to 0.35-0.60
            scaled = 0.35 + (pred / y.max()) * 0.25
        elif tier == 2:  # Mid tier
            # Scale to 0.20-0.45
            scaled = 0.20 + (pred / y.max()) * 0.25
        else:  # Bottom tier
            # Scale to 0.05-0.30
            scaled = 0.05 + (pred / y.max()) * 0.25
        
        # Add some randomness to avoid identical values
        jitter = np.random.uniform(-0.03, 0.03)
        final_prob = np.clip(scaled + jitter, min_prob, max_prob)
        scaled_predictions.append(final_prob)
    
    # Add predictions to dataframe
    gw31_data["predicted_clean_sheets"] = scaled_predictions
    
    # 13. Sort and save results
    result_df = gw31_data.sort_values(by="predicted_clean_sheets", ascending=False)
    
    # Check distribution of predictions
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    hist, _ = np.histogram(result_df["predicted_clean_sheets"], bins=bins)
    
    print("Distribution of clean sheet probabilities:")
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} predictions")
    
    # Save only needed columns
    out_path = "GW31_Predicted_Clean_Sheets.csv"  # Updated filename
    result_df[["name", "team", "GW", "predicted_clean_sheets"]].to_csv(out_path, index=False)
    
    print(f"Saved predictions to {out_path}")
    print("Top 10 players most likely to get clean sheets:")
    print(result_df[["name", "team", "predicted_clean_sheets"]].head(10))

if __name__ == "__main__":
    main()