"""
FPL Goal Prediction Model with Fixture Difficulty

This script predicts goals for Fantasy Premier League players by combining:
1. Player-specific historical performance metrics
2. Team strength data from the FPL API  
3. Fixture difficulty adjustments
4. Position-based baseline predictions

"""
import pandas as pd  # Data manipulation and analysis 
import numpy as np  # Numerical Operations  
from sklearn.model_selection import train_test_split  # Creating validation datasets
from sklearn.ensemble import RandomForestRegressor  # Main Prediction algorithm Used RF because it 
from sklearn.metrics import mean_absolute_error  # Model Evaluation
from sklearn.impute import SimpleImputer  # For handling missing values
import os  # File paths
import matplotlib.pyplot as plt  # Visualizations
from pathlib import Path

# New function to enhance data with team strength metrics
def enhance_with_team_data(data, teams_df):
    """
    Add team strength metrics to player data without needing external ID mapping

    Why we need this:
    - Player performance is heavily influenced by their team's attacking strength
    - Opponent defensive strength significantly impacts goal probability
    - Home/away performance varies significantly for teams
    """
    team_name_to_id = {team['name']: team['id'] for _, team in teams_df.iterrows()}
    team_strength = {
        team['id']: {
            'def_home': team['strength_defence_home'],
            'def_away': team['strength_defence_away'],
            'atk_home': team['strength_attack_home'],
            'atk_away': team['strength_attack_away'],
            'overall_home': team['strength_overall_home'],
            'overall_away': team['strength_overall_away']
        } for _, team in teams_df.iterrows()
    }

    enhanced_data = data.copy()

    enhanced_data['team_attacking'] = enhanced_data.apply(
        lambda row: team_strength.get(team_name_to_id.get(row['team'], 0), {}).get('atk_home' if row['was_home'] else 'atk_away', 1100), axis=1)
    enhanced_data['team_overall'] = enhanced_data.apply(
        lambda row: team_strength.get(team_name_to_id.get(row['team'], 0), {}).get('overall_home' if row['was_home'] else 'overall_away', 1100), axis=1)
    enhanced_data['opp_defending'] = enhanced_data.apply(
        lambda row: team_strength.get(row['opponent_team'], {}).get('def_home' if not row['was_home'] else 'def_away', 1100), axis=1)
    enhanced_data['opp_overall'] = enhanced_data.apply(
        lambda row: team_strength.get(row['opponent_team'], {}).get('overall_home' if not row['was_home'] else 'overall_away', 1100), axis=1)
    enhanced_data['fixture_difficulty'] = enhanced_data.apply(
        lambda row: ((row['team_attacking'] - row['opp_defending']) + (row['team_overall'] - row['opp_overall'])) / 100, axis=1)

    return enhanced_data

# Modified function to use dynamic GW
def prepare_upcoming_gw_data(all_players, player_avg_stats, teams_df, fixtures_df, current_gw):
    """
    Prepare prediction data for the specified Gameweek with team strength metrics.
    """
    upcoming_data = all_players.merge(player_avg_stats, on=["name", "team"], how="left")
    upcoming_data["GW"] = current_gw

    upcoming_fixtures = fixtures_df[fixtures_df['event'] == current_gw].copy()
    team_id_to_name = {team['id']: team['name'] for _, team in teams_df.iterrows()}
    team_name_to_id = {team['name']: team['id'] for _, team in teams_df.iterrows()}

    def get_opponent_and_home(row):
        team_id = team_name_to_id.get(row['team'])
        if team_id is None:
            return None, False, "Unknown"

        fixture = upcoming_fixtures[
            (upcoming_fixtures['team_h'] == team_id) | (upcoming_fixtures['team_a'] == team_id)  # Fixed: Correct syntax
        ]

        if fixture.empty:
            return None, False, "Unknown"

        fixture = fixture.iloc[0]
        if fixture['team_h'] == team_id:
            opponent_id = fixture['team_a']
            return opponent_id, True, team_id_to_name.get(opponent_id, "Unknown")
        else:
            opponent_id = fixture['team_h']
            return opponent_id, False, team_id_to_name.get(opponent_id, "Unknown")

    upcoming_data[['opponent_team', 'was_home', 'opponent_name']] = pd.DataFrame(
        upcoming_data.apply(get_opponent_and_home, axis=1).tolist(),
        index=upcoming_data.index
    )

    return enhance_with_team_data(upcoming_data, teams_df)

# Updated to use current_gw instead of hardcoded GW31
def prepare_gw_data(all_players, player_avg_stats, teams_df, fixtures_df, current_gw):
    """
    Prepare prediction data for the specified Gameweek with team strength metrics.
    
    Why we need this:
    - To predict future performance, we need to know who each player is facing
    - Players' historical averages need to be combined with upcoming fixture data
    - FPL API stores fixtures separately from player data, requiring this integration step
    """
    gw_data = all_players.merge(player_avg_stats, on=["name", "team"], how="left")
    gw_data["GW"] = current_gw
    
    gw_fixtures = fixtures_df[fixtures_df['event'] == current_gw].copy()
    
    team_id_to_name = {team['id']: team['name'] for _, team in teams_df.iterrows()}
    team_name_to_id = {team['name']: team['id'] for _, team in teams_df.iterrows()}
    
    def get_opponent_and_home(row):
        team_id = team_name_to_id.get(row['team'])
        if team_id is None:
            return None, False, "Unknown"
            
        fixture = gw_fixtures[
            (gw_fixtures['team_h'] == team_id) | 
            (gw_fixtures['team_a'] == team_id)
        ]
        
        if fixture.empty:
            return None, False, "Unknown"
            
        fixture = fixture.iloc[0]
        
        if fixture['team_h'] == team_id:
            opponent_id = fixture['team_a']
            opponent_name = team_id_to_name.get(opponent_id, "Unknown")
            return opponent_id, True, opponent_name
        else:
            opponent_id = fixture['team_h']
            opponent_name = team_id_to_name.get(opponent_id, "Unknown")
            return opponent_id, False, opponent_name
    
    gw_data[['opponent_team', 'was_home', 'opponent_name']] = pd.DataFrame(
        gw_data.apply(get_opponent_and_home, axis=1).tolist(),
        index=gw_data.index
    )
    
    gw_data = enhance_with_team_data(gw_data, teams_df)
    
    return gw_data

def main():
    BASE = Path(__file__).resolve().parents[1]

    # 1) Load master player-GW data
    DATA_PATH = BASE / "data" / "processed" / "merged_gw_cleaned.csv"
    data = pd.read_csv(DATA_PATH, on_bad_lines="skip")
    print(f"✅ Loaded player-GW data: {data.shape}")

    # 2) Load teams & fixtures
    teams_path = BASE / "data" / "processed" / "teams.csv"
    fixtures_path = BASE / "data" / "processed" / "fixtures.csv"
    teams_df = pd.read_csv(teams_path)
    fixtures_df = pd.read_csv(fixtures_path, parse_dates=["kickoff_time"])
    
    # Ensure kickoff_time is UTC-aware
    if fixtures_df["kickoff_time"].dt.tz is None:
        fixtures_df["kickoff_time"] = fixtures_df["kickoff_time"].dt.tz_localize('UTC')

    print(f"✅ Loaded teams     : {teams_path.name}")
    print(f"✅ Loaded fixtures  : {fixtures_path.name}")

    # 3) Latest completed GW by date
    now = pd.Timestamp.now(tz='UTC')
    completed = fixtures_df[fixtures_df["kickoff_time"] <= now]
    latest_completed_gw = int(completed["event"].max())
    latest_gw = latest_completed_gw + 1

    print(f"📆 Latest completed GW: GW{latest_completed_gw}")
    print(f"🔮 Predicting upcoming : GW{latest_gw}")

    # 4) Build train/test up through GW32
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= min(latest_completed_gw, 32))]
    print(f"Training on GW1–GW{min(latest_completed_gw, 32)}: {train_data.shape}")

    # 5) Slice next-GW fixtures
    gw_fixtures = fixtures_df[fixtures_df["event"] == latest_gw]
    print(f"Found {len(gw_fixtures)} fixtures for GW{latest_gw}")
    
    # Print data overview for debugging
    print("Columns in merged_gw_cleaned.csv:", data.columns.tolist())
    print("Sample rows:\n", data.head())

    # Verify training data availability
    if train_data.empty:
        print(f"No training data found for GW1–GW{min(latest_completed_gw, 32)} in merged_gw_cleaned.csv.")
        return

    print(f"Training data shape (GW1–GW{min(latest_completed_gw, 32)}): {train_data.shape}")

    # Check for position data
    has_position_data = 'position' in train_data.columns
    print(f"Position data available: {has_position_data}")
    
    # Create position-specific features
    if has_position_data:
        train_data['is_forward'] = (train_data['position'] == 'FWD').astype(int)
        train_data['is_midfielder'] = (train_data['position'] == 'MID').astype(int) 
        train_data['is_defender'] = (train_data['position'] == 'DEF').astype(int)
        train_data['is_goalkeeper'] = (train_data['position'] == 'GK').astype(int)
    else:
        train_data['is_forward'] = 0
        train_data['is_midfielder'] = 0
        train_data['is_defender'] = 0
        train_data['is_goalkeeper'] = 0

    # Enhance training data with team strength metrics
    train_data = enhance_with_team_data(train_data, teams_df)
    print("Added team strength and fixture difficulty metrics to training data")

    # Select features & target
    features = [
        "minutes",
        "xP",
        "threat",
        "transfers_in",
        "expected_goals",
        "creativity",
        "ict_index",
        "starts",
        "is_forward",
        "is_midfielder",
        "is_defender",
        "is_goalkeeper",
        "team_attacking",
        "opp_defending",
        "fixture_difficulty"
    ]

    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        features = [f for f in features if f in train_data.columns]
        print(f"Using these features instead: {features}")

    target = "goals_scored"

    # Convert columns to numeric
    for col in features + [target]:
        if col in train_data.columns:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # Handle missing target values
    train_data_with_target = train_data.dropna(subset=[target])
    
    # Calculate average goals by position
    position_goals = {}
    if has_position_data:
        position_goals = train_data_with_target.groupby('position')[target].mean().to_dict()
        print("Average goals by position:", position_goals)
    else:
        position_goals = {'FWD': 0.15, 'MID': 0.08, 'DEF': 0.03, 'GK': 0.001, 'Unknown': 0.05}
    
    # Impute missing feature values
    imputer = SimpleImputer(strategy='mean')
    X_train_with_target = train_data_with_target[features].copy()
    
    for col in X_train_with_target.columns:
        if X_train_with_target[col].isna().any():
            X_train_with_target[col] = X_train_with_target[col].fillna(0)
    
    imputer.fit(X_train_with_target)
    X_train_imputed = imputer.transform(X_train_with_target)
    X_train_df = pd.DataFrame(X_train_imputed, columns=features, index=train_data_with_target.index)
    y_train = train_data_with_target[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate model
    predictions_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print("Mean Absolute Error (MAE) on the test set:", mae)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # Get unique players
    all_players = data[["name", "team"]].drop_duplicates()
    
    if has_position_data:
        player_positions = data.groupby(["name", "team"])['position'].first().reset_index()
        all_players = all_players.merge(player_positions, on=["name", "team"], how="left")
    else:
        all_players['position'] = 'Unknown'
        
    print(f"Total unique players in dataset: {len(all_players)}")

    # Prepare player stats
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    player_avg_stats = train_data.groupby(["name", "team"])[numeric_columns].mean().reset_index()
    
    # Prepare GW data for prediction
    gw_data = prepare_gw_data(all_players, player_avg_stats, teams_df, fixtures_df, latest_gw)
    print(f"Prepared GW{latest_gw} data with fixture information for {len(gw_data)} players")
    
    if 'roll3_minutes' not in gw_data.columns:
        gw_data['roll3_minutes'] = gw_data['minutes'] 

    gw_data = gw_data[gw_data['opponent_name'] != 'Unknown']
    gw_data = gw_data[gw_data['roll3_minutes'] >= 30]

    # Create position-specific features
    gw_data['is_forward'] = (gw_data['position'] == 'FWD').astype(int)
    gw_data['is_midfielder'] = (gw_data['position'] == 'MID').astype(int) 
    gw_data['is_defender'] = (gw_data['position'] == 'DEF').astype(int)
    gw_data['is_goalkeeper'] = (gw_data['position'] == 'GK').astype(int)
    
    # Impute missing values
    X_future = gw_data[features].copy()
    
    for col in X_future.columns:
        if X_future[col].isna().any():
            X_future[col] = X_future[col].fillna(0)
    
    X_future_imputed = imputer.transform(X_future)
    X_future_df = pd.DataFrame(X_future_imputed, columns=features)
    
    # Get predictions
    raw_predictions = model.predict(X_future_df)
    
    def get_min_probability(row):
        position = row.get('position', 'Unknown')
        minutes_avg = row.get('minutes', 0)
        if pd.isna(minutes_avg):
            minutes_avg = 0
            
        if position == 'FWD':
            base_min = 0.20
        elif position == 'MID':
            base_min = 0.12
        elif position == 'DEF':
            base_min = 0.06
        elif position == 'GK':
            base_min = 0.01
        else:
            base_min = 0.10
        
        minutes_factor = min(1.0, minutes_avg / 60)
        min_prob = base_min * max(0.2, minutes_factor)
        
        return min_prob
    
    # Apply smoothing
    alpha = 0.65
    gw_data['position_avg_goals'] = gw_data['position'].map(
        lambda pos: position_goals.get(pos, 0.05)
    )
    
    gw_data['raw_predicted_goals'] = raw_predictions
    gw_data['min_probability'] = gw_data.apply(get_min_probability, axis=1)

    def apply_fixture_adjustment(row):
        raw_prediction = row['raw_predicted_goals']
        fixture_diff = row['fixture_difficulty']
        fixture_scale = 0.60
        normalized_fixture = (fixture_diff + 5) / 10
        normalized_fixture = max(0, min(1, normalized_fixture))
        fixture_adjustment = (normalized_fixture - 0.5) * fixture_scale * 2
        adjusted_prediction = raw_prediction + fixture_adjustment
        return max(0, adjusted_prediction)
    
    gw_data['fixture_adjusted_goals'] = gw_data.apply(apply_fixture_adjustment, axis=1)
    
    # Print fixture adjustment examples
    fixture_examples = gw_data.sort_values('fixture_difficulty').sample(min(10, len(gw_data)))
    adjustment_examples = gw_data.sort_values('fixture_difficulty')[['name', 'team', 'position', 'opponent_name', 
                           'fixture_difficulty', 'raw_predicted_goals', 'fixture_adjusted_goals']]
    
    print("\nPlayers with tough fixtures (showing raw vs adjusted predictions):")
    print(adjustment_examples.head(5).to_string(index=False))
    
    print("\nPlayers with favorable fixtures (showing raw vs adjusted predictions):")
    print(adjustment_examples.tail(5).to_string(index=False))
    
    print("\nSample of fixture adjustments:")
    print(gw_data.sample(min(10, len(gw_data)))[['name', 'team', 'raw_predicted_goals', 'fixture_difficulty', 'fixture_adjusted_goals']].sort_values('fixture_difficulty'))
    
    # Blend predictions
    gw_data['blended_prediction'] = (alpha * gw_data['fixture_adjusted_goals']) + ((1-alpha) * gw_data['position_avg_goals'])
    gw_data['predicted_goals'] = gw_data.apply(
        lambda row: max(row['blended_prediction'], row['min_probability']), axis=1
    )
    
    gw_data['predicted_goals'] = gw_data['predicted_goals'].round(2)
    gw_data_sorted = gw_data.sort_values(by="predicted_goals", ascending=False)
    
    # Prediction statistics
    print("\nDistribution of predicted goal values:")
    prediction_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0]
    prediction_counts = pd.cut(gw_data_sorted["predicted_goals"], bins=prediction_bins, include_lowest=True).value_counts().sort_index()
    print(prediction_counts)
    
    # Save predictions
    out_path = f"GW{latest_gw}_Predicted_goals_with_fixtures.csv"
    columns_to_save = ["name", "team", "position", "GW", "opponent_name", "was_home", 
                      "fixture_difficulty", "predicted_goals", "raw_predicted_goals", 
                      "fixture_adjusted_goals", "blended_prediction", "min_probability",
                      "opponent_team", "team_attacking", "opp_defending"]
    columns_available = [col for col in columns_to_save if col in gw_data_sorted.columns]
    gw_data_sorted[columns_available].to_csv(out_path, index=False)
    print(f"Sorted predictions for GW{latest_gw} saved to {out_path}")
    print("Sample predictions with fixture data:\n", 
          gw_data_sorted[["name", "team", "position", "opponent_name", "was_home", "fixture_difficulty", "raw_predicted_goals", "fixture_adjusted_goals", "predicted_goals"]].head(10))
    
    # Save distribution
    distribution_path = f"GW{latest_gw}_Prediction_Distribution_with_fixtures.csv"
    prediction_counts.to_csv(distribution_path)
    print(f"Prediction distribution saved to {distribution_path}")
    
    # Create visualization
    try:
        plt.figure(figsize=(12, 6))
        plt.hist(gw_data_sorted["predicted_goals"], bins=prediction_bins, alpha=0.7, color='royalblue')
        plt.title(f'Distribution of Predicted Goal Probabilities for GW{latest_gw} (With Fixture Difficulty)')
        plt.xlabel('Predicted Goals')
        plt.ylabel('Number of Players')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f'GW{latest_gw}_Goal_Distribution_with_fixtures.png')
        plt.close()
        print(f"Distribution visualization saved to GW{latest_gw}_Goal_Distribution_with_fixtures.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Create fixture adjustment visualization
    try:
        plt.figure(figsize=(12, 8))
        plt.scatter(gw_data["fixture_difficulty"], gw_data["raw_predicted_goals"], 
                   alpha=0.3, color='blue', label='Raw Predictions')
        plt.scatter(gw_data["fixture_difficulty"], gw_data["fixture_adjusted_goals"], 
                   alpha=0.3, color='red', label='Fixture-Adjusted Predictions')
        
        x_range = np.linspace(gw_data["fixture_difficulty"].min(), gw_data["fixture_difficulty"].max(), 100)
        z_raw = np.polyfit(gw_data["fixture_difficulty"], gw_data["raw_predicted_goals"], 1)
        p_raw = np.poly1d(z_raw)
        plt.plot(x_range, p_raw(x_range), '--', color='blue', 
                label=f'Raw Trend (slope: {z_raw[0]:.3f})')
        
        z_adj = np.polyfit(gw_data["fixture_difficulty"], gw_data["fixture_adjusted_goals"], 1)
        p_adj = np.poly1d(z_adj)
        plt.plot(x_range, p_adj(x_range), '-', color='red', 
                label=f'Adjusted Trend (slope: {z_adj[0]:.3f})')
        
        raw_slope = z_raw[0]
        adjusted_slope = z_adj[0]
        impact_increase = ((adjusted_slope / raw_slope) - 1) * 100 if raw_slope != 0 else float('inf')
        
        plt.title(f'Impact of Fixture Difficulty on Predictions\nFixture impact increased by {impact_increase:.1f}%')
        plt.xlabel('Fixture Difficulty (higher = easier fixture)')
        plt.ylabel('Predicted Goals')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Fixture_Adjustment_Impact.png')
        plt.close()
        print("Fixture adjustment impact visualization saved to Fixture_Adjustment_Impact.png")
        
        plt.figure(figsize=(12, 6))
        plt.hist(gw_data["raw_predicted_goals"], bins=20, alpha=0.5, color='blue', label='Raw Predictions')
        plt.hist(gw_data["fixture_adjusted_goals"], bins=20, alpha=0.5, color='red', label='Fixture-Adjusted')
        plt.title('Distribution of Goal Predictions Before and After Fixture Adjustment')
        plt.xlabel('Predicted Goals')
        plt.ylabel('Number of Players')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Prediction_Distribution_Comparison.png')
        plt.close()
        print("Prediction distribution comparison saved to Prediction_Distribution_Comparison.png")
        
    except Exception as e:
        print(f"Could not create fixture adjustment visualization: {e}")

if __name__ == "__main__":
    main()