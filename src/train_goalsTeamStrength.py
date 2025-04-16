"""
FPL Goal Prediction Model with Fixture Difficulty

This script predicts goals for Fantasy Premier League players by combining:
1. Player-specific historical performance metrics
2. Team strength data from the FPL API  
3. Fixture difficulty adjustments
4. Position-based baseline predictions

Author: Dylan Chai
"""
import pandas as pd # Data maniulation and analysis 
import numpy as np  # Numerical Operations  
from sklearn.model_selection import train_test_split    #creating validation datasets
from sklearn.ensemble import RandomForestRegressor  # Main Prediction algorithm Used RF because it 
from sklearn.metrics import mean_absolute_error # Model Evaluation
from sklearn.impute import SimpleImputer    # For handling missing balues
import os   # File paths
import matplotlib.pyplot as plt # Visulisations

# These imports are for data manipulation machine learning and some visualtion.
# Below is the train goals model with team strength added, this plays a crucial role in predictions
# As on a weekly basis fixtures change, before the implementation of this predictions looked more like averages
# Now with team strength players with decent fixtures now appear even if they dont have 
# Amazing weeks prior

# New function to enhance data with team strength metrics
def enhance_with_team_data(data, teams_df):
    """
    Add team strength metrics to player data without needing external ID mapping

      Why we need this:
    - Player performance is heavily influenced by their team's attacking strength
    - Opponent defensive strength significantly impacts goal probability
    - Home/away performance varies significantly for teams

    """
    # Create team name to ID mapping - needed because the data uses team names but the API uses IDs
    team_name_to_id = {}
    for _, team in teams_df.iterrows():
        team_name_to_id[team['name']] = team['id']
    
    # Create team ID to strength metrics mapping
    team_strength = {}
    for _, team in teams_df.iterrows():
        team_strength[team['id']] = {
            'def_home': team['strength_defence_home'],
            'def_away': team['strength_defence_away'],
            'atk_home': team['strength_attack_home'],
            'atk_away': team['strength_attack_away'],
            'overall_home': team['strength_overall_home'],
            'overall_away': team['strength_overall_away']
        }
    
    # Create a copy to avoid SettingWithCopyWarning - pandas issues warnings when modifying views of dataframes
    enhanced_data = data.copy()
    
    # Add team attacking strength (for player's team) players who player for stronger teams score more goals 
    enhanced_data['team_attacking'] = enhanced_data.apply(
        lambda row: team_strength.get(
            team_name_to_id.get(row['team'], 0), {}
        ).get('atk_home' if row['was_home'] else 'atk_away', 1100),
        axis=1
    )
    
    # Add team overall strength - overall strength affects overall performance 
    enhanced_data['team_overall'] = enhanced_data.apply(
        lambda row: team_strength.get(
            team_name_to_id.get(row['team'], 0), {}
        ).get('overall_home' if row['was_home'] else 'overall_away', 1100),
        axis=1
    )
    
    # Add opponent defensive strength because weaker defenses code more goals
    enhanced_data['opp_defending'] = enhanced_data.apply(
        lambda row: team_strength.get(
            row['opponent_team'], {}
        ).get('def_home' if not row['was_home'] else 'def_away', 1100),
        axis=1
    )
    
    # Add opponent overall strength a general metric to indicate team quality 
    enhanced_data['opp_overall'] = enhanced_data.apply(
        lambda row: team_strength.get(
            row['opponent_team'], {}
        ).get('overall_home' if not row['was_home'] else 'overall_away', 1100),
        axis=1
    )
    

    # Calculate fixture difficulty - higher values mean easier fixtures for goal scoring
    # We use the difference between attacking and defensive strengths because this directly impacts scoring probability
    # Scaling by 100 makes the values more intuitive 
    enhanced_data['fixture_difficulty'] = enhanced_data.apply(
        lambda row: (
            (row['team_attacking'] - row['opp_defending']) + 
            (row['team_overall'] - row['opp_overall'])
        ) / 100, 
        axis=1
    )
    
    return enhanced_data

# To prepare GW31 data with fixture information
def prepare_gw31_data(all_players, player_avg_stats, teams_df, fixtures_df):
    """
Prepare GW31 prediction data with team strength metrics.
    
    Why we need this:
    - To predict future performance, we need to know who each player is facing
    - Players historical averages need to be combined with upcoming fixture data
    - FPL API stores fixtures separately from player data, requiring this integration step

    """
    # Merge player data with their historical averages - we use left join to keep all players even if they have limited data
    gw31_data = all_players.merge(player_avg_stats, on=["name", "team"], how="left")
    gw31_data["GW"] = 31
    
    # Get GW31 fixtures
    gw31_fixtures = fixtures_df[fixtures_df['event'] == 31].copy()
    
    # Create team name to ID mapping
    team_id_to_name = {}
    team_name_to_id = {}
    
    for _, team in teams_df.iterrows():
        team_id_to_name[team['id']] = team['name']
        team_name_to_id[team['name']] = team['id']
    
    # Define function to find opponent and home/away status for each player
    def get_opponent_and_home(row):
        team_id = team_name_to_id.get(row['team'])
        if team_id is None:
            return None, False, "Unknown" # if a team cannot be mapped
            
        # Find fixture where this team plays - using logical OR to find either home or away matches
        fixture = gw31_fixtures[
            (gw31_fixtures['team_h'] == team_id) | 
            (gw31_fixtures['team_a'] == team_id)
        ]
        
        if fixture.empty:
            return None, False, "Unknown"
            
        fixture = fixture.iloc[0] # Get the first match if multiple exist
        
        if fixture['team_h'] == team_id:
            # Team is home
            opponent_id = fixture['team_a']
            opponent_name = team_id_to_name.get(opponent_id, "Unknown")
            return opponent_id, True, opponent_name
        else:
            # Team is away
            opponent_id = fixture['team_h']
            opponent_name = team_id_to_name.get(opponent_id, "Unknown")
            return opponent_id, False, opponent_name
    
    # Apply function to get opponent and home status - using DataFrame to efficiently handle multiple return values
    gw31_data[['opponent_team', 'was_home', 'opponent_name']] = pd.DataFrame(
        gw31_data.apply(get_opponent_and_home, axis=1).tolist(),
        index=gw31_data.index
    )
    
    # Enhance with team strength data after opponent information is added
    gw31_data = enhance_with_team_data(gw31_data, teams_df)
    
    return gw31_data

def main():
        # Construct path to the merged gameweek data
    # Using os.path.join ensures path separators are correct across operating systems
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "external", "Fantasy-Premier-League", "data", "2024-25", "gws", "merged_gw_cleaned.csv")

    # Check if file exists before attempting to load it - provides a clearer error message
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        print("Please run the CSV cleaner script first to create the cleaned CSV file.")
        return
      # Use try/except to handle potential errors when loading data
    # This makes my code more robust when dealing with file operations
    try:
        data = pd.read_csv(data_path, low_memory=False) # low_memory=False prevents mixed data type warnings
        print(f"Successfully loaded data with shape: {data.shape}")
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return

    # Load team data for strength metrics
    teams_path = os.path.join(base_dir, "..", "external", "Fantasy-Premier-League", "data", "2024-25", "teams.csv")
    if not os.path.exists(teams_path):
        print(f"Team data not found: {teams_path}")
        return
    
    teams_df = pd.read_csv(teams_path)
    print(f"Successfully loaded team data with {len(teams_df)} teams")
    
    # Load fixtures data for GW31 - needed for predicting future gameweek
    fixtures_path = os.path.join(base_dir, "..", "external", "Fantasy-Premier-League", "data", "2024-25", "fixtures.csv")
    if not os.path.exists(fixtures_path):
        print(f"Fixtures data not found: {fixtures_path}")
        return
    
    fixtures_df = pd.read_csv(fixtures_path)
    gw31_fixtures = fixtures_df[fixtures_df['event'] == 31]
    print(f"Found {len(gw31_fixtures)} fixtures for GW31")

    # Print data overview to help with debugging and understanding the dataset

    print("Columns in merged_gw_cleaned.csv:", data.columns.tolist())
    print("Sample rows:\n", data.head())

    # This prevents using future data that wouldn't be available in a real prediction scenario
    
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 30)]

    # Verify data availability to fail gracefully if no data is found
    if train_data.empty:
        print("No training data found for GW 1-30 in merged_gw_cleaned.csv.")
        return

    print(f"Training data shape (GW 1-30): {train_data.shape}")

    # Check if position data is available
    has_position_data = 'position' in train_data.columns
    print(f"Position data available: {has_position_data}")
    
    # Create position-specific features as one-hot encodings
    # This allows the model to learn different patterns for each position
    # One-hot encoding is used because positions are categorical variables

    if has_position_data:
        # One-hot encode positions to use as features
        train_data['is_forward'] = (train_data['position'] == 'FWD').astype(int)
        train_data['is_midfielder'] = (train_data['position'] == 'MID').astype(int) 
        train_data['is_defender'] = (train_data['position'] == 'DEF').astype(int)
        train_data['is_goalkeeper'] = (train_data['position'] == 'GK').astype(int)
    else:
        # If position data not available, add placeholder columns
        train_data['is_forward'] = 0
        train_data['is_midfielder'] = 0
        train_data['is_defender'] = 0
        train_data['is_goalkeeper'] = 0

     # Enhance training data with team strength metrics
    # This is crucial because player performance is highly dependent on team context
    train_data = enhance_with_team_data(train_data, teams_df)
    print("Added team strength and fixture difficulty metrics to training data")

     # Select features & target
    # Each feature was chosen based on its relevance to goal scoring probability:
    features = [
       "minutes",           # How long the player played - more minutes = more scoring opportunities
        "xP",                # Expected points from FPL API - contains FPL's own prediction information
        "threat",            # Attacking threat metric - directly relates to goal scoring probability
        # "bps" removed because bonus points are calculated after goals, causing data leakage
        "transfers_in",      # Proxy for form and player popularity - correlates with recent good performances
        "expected_goals",    # Direct statistical measure of goal expectation - highest predictive power
        "creativity",        # Measures chance creation - correlates with attacking involvement
        "ict_index",         # Composite of influence, creativity, and threat - broad performance indicator
        "starts",            # Starting players have more chances to score than substitutes
        "is_forward",        # Position features - forwards score more than other positions
        "is_midfielder",     # Midfielders score more than defenders but less than forwards
        "is_defender",       # Defenders occasionally score, especially on set pieces
        "is_goalkeeper",     # Goalkeepers rarely score
        # Team strength features provide contextual information about match difficulty
        "team_attacking",    # Player's team attacking strength - stronger attack means more goals
        "opp_defending",     # Opponent's defensive strength - weaker defenses concede more
        "fixture_difficulty" # Combined fixture difficulty rating - summarizes match-up quality
    ]

    # Check for missing features and adapt the model accordingly
    # This makes the code robust to different data formats or missing columns
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        # Remove missing features from the list
        features = [f for f in features if f in train_data.columns]
        print(f"Using these features instead: {features}")

    target = "goals_scored" # what im trying to predict

    # Convert columns to numeric if needed - ensures consistent data types
    # Using errors='coerce' converts non-numeric values to NaN rather than crashing
    for col in features + [target]:
        if col in train_data.columns:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # Create a copy of training data with complete target values for model training
    # We drop rows with missing target vlaues because these cant be used for training 
    train_data_with_target = train_data.dropna(subset=[target])
    
    # Calculate average goals by position for baseline predictions
    position_goals = {}
    if has_position_data:
        position_goals = train_data_with_target.groupby('position')[target].mean().to_dict()
        print("Average goals by position:", position_goals)
    else:
        # Default values if position data not available
        position_goals = {'FWD': 0.15, 'MID': 0.08, 'DEF': 0.03, 'GK': 0.001, 'Unknown': 0.05}
    
    #Use imputation instead of dropping rows with missing feature values -     # This preserves more training data and makes predictions more robust

    imputer = SimpleImputer(strategy='mean')
    
    # Fit the imputer on the training data
    X_train_with_target = train_data_with_target[features].copy()
    
    # Handle any remaining NaN values for imputation
    # We first fill with zeros as a safety measure, then apply the imputer
    for col in X_train_with_target.columns:
        if X_train_with_target[col].isna().any():
            X_train_with_target[col] = X_train_with_target[col].fillna(0)
    
    imputer.fit(X_train_with_target)
    
    # Transform the feature data for training
    X_train_imputed = imputer.transform(X_train_with_target)
    
    # Convert back to DataFrame for clarity
    X_train_df = pd.DataFrame(X_train_imputed, columns=features, index=train_data_with_target.index)
    y_train = train_data_with_target[target]

    # Split the imputed training data into train/test sets for evaluation
    # Using 80/20 split 
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor model with optimized parameters
    # RF was chosen becyase of:
    # Ability to capture non linear relationships
    # Robust handling of different feature scales
    # good performace with limited data 
    # automaitc feature selection through tree splitting
    model = RandomForestRegressor(
        n_estimators=200,           # More trees for better probability distribution
        min_samples_leaf=5,         # Increase to prevent overfitting/extreme predictions 5 smaples per leaf
        max_features='sqrt',        # Standard practice for random forests
        random_state=42,
        n_jobs=-1                   # Use all available cores
    )
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print("Mean Absolute Error (MAE) on the test set:", mae)

    # Feature importance analysis
    # This tells us what drives predictions this validates out feature selection giving insight
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # Get all unique players from the dataset for prediction
    all_players = data[["name", "team"]].drop_duplicates()
    
    # Get position data if available
    if has_position_data:
        # Get the most common position for each player
        player_positions = data.groupby(["name", "team"])['position'].first().reset_index()
        all_players = all_players.merge(player_positions, on=["name", "team"], how="left")
    else:
        # Add placeholder position column
        all_players['position'] = 'Unknown'
        
    print(f"Total unique players in dataset: {len(all_players)}")

    # Prepare player stats for GW31 prediction by averaging their previous performances
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    player_avg_stats = train_data.groupby(["name", "team"])[numeric_columns].mean().reset_index()
    
    # Prepare GW31 data with fixture information
        # This combines historical player performance with upcoming fixture data
    gw31_data = prepare_gw31_data(all_players, player_avg_stats, teams_df, fixtures_df)
    print(f"Prepared GW31 data with fixture information for {len(gw31_data)} players")
    
    # Create position-specific features for prediction
    # Handle position-based features
    gw31_data['is_forward'] = (gw31_data['position'] == 'FWD').astype(int)
    gw31_data['is_midfielder'] = (gw31_data['position'] == 'MID').astype(int) 
    gw31_data['is_defender'] = (gw31_data['position'] == 'DEF').astype(int)
    gw31_data['is_goalkeeper'] = (gw31_data['position'] == 'GK').astype(int)
    
    # Impute any missing values for prediction
    X_future = gw31_data[features].copy()
    
    # Handle any remaining NaN values before imputation
    for col in X_future.columns:
        if X_future[col].isna().any():
            X_future[col] = X_future[col].fillna(0)
    
    X_future_imputed = imputer.transform(X_future)
    X_future_df = pd.DataFrame(X_future_imputed, columns=features)
    
    # Get raw model predictions
    raw_predictions = model.predict(X_future_df)
    
    # Function to get position-based minimum probability - MORE AGGRESSIVE
    def get_min_probability(row):
        position = row.get('position', 'Unknown')
        minutes_avg = row.get('minutes', 0)
        if pd.isna(minutes_avg):
            minutes_avg = 0
            
        # Base minimum probability by position - 
        if position == 'FWD':
            base_min = 0.20  # Much higher floor for forwards
        elif position == 'MID':
            base_min = 0.12  # Higher floor for midfielders
        elif position == 'DEF':
            base_min = 0.06  # Higher floor for defenders
        elif position == 'GK':
            base_min = 0.01  # Slightly higher floor for goalkeepers
        else:
            base_min = 0.10  # Higher default
        
        # Scale by average minutes - MORE GENEROUS SCALING
        minutes_factor = min(1.0, minutes_avg / 60)  # Scale fully at 60 minutes instead of 90
        min_prob = base_min * max(0.2, minutes_factor)  # Higher minimum factor
        
        return min_prob
    
    # Apply a more realistic smoothing to prediction values
    # This avoids the sharp cutoff between meaningful and minimum values
    alpha = 0.5  # Balanced blending factor (50/50 split)
    
    # Add the position-based average goals as a new column
    # Map the position to average goals, with a fallback value
    gw31_data['position_avg_goals'] = gw31_data['position'].map(
        lambda pos: position_goals.get(pos, 0.05)
    )
    
    # Blend model predictions with position averages and apply minimum probabilities
    gw31_data['raw_predicted_goals'] = raw_predictions
    gw31_data['min_probability'] = gw31_data.apply(get_min_probability, axis=1)
    
    # Apply a consistent fixture adjustment to all players
    def apply_fixture_adjustment(row):
        # Get base values
        raw_prediction = row['raw_predicted_goals']
        fixture_diff = row['fixture_difficulty']
        
        # FIXTURE IMPORTANCE Adjust this single value to control fixture impact
        # Higher values will increase the impact of fixtures on predictions
        # 0.25 means fixtures can impact predictions by up to +-25 percentage points
        fixture_scale = 0.50
        
        # Normalize fixture difficulty to a 0-1 range
        # Fixtures range from roughly -5 to +5, so we add 5 and divide by 10
        normalized_fixture = (fixture_diff + 5) / 10
        
        # Constrain to 0-1 range in case of extreme fixture values
        normalized_fixture = max(0, min(1, normalized_fixture))
        
        # Calculate the fixture adjustment (-fixture_scale to +fixture_scale)
        fixture_adjustment = (normalized_fixture - 0.5) * fixture_scale * 2
        
        # Apply the adjustment to the raw prediction
        adjusted_prediction = raw_prediction + fixture_adjustment
        
        # Ensure non-negative
        return max(0, adjusted_prediction)
    
    # Apply fixture adjustment
    gw31_data['fixture_adjusted_goals'] = gw31_data.apply(apply_fixture_adjustment, axis=1)
    
    # Print before/after examples sorted by fixture difficulty
    fixture_examples = gw31_data.sort_values('fixture_difficulty').sample(min(10, len(gw31_data)))
    
    print("\nFixture Adjustment Examples (sorted by fixture difficulty):")
    adjustment_examples = gw31_data.sort_values('fixture_difficulty')[['name', 'team', 'position', 'opponent_name', 
                           'fixture_difficulty', 'raw_predicted_goals', 'fixture_adjusted_goals']]
    
    # Show examples of very good and very bad fixtures
    bad_fixtures = adjustment_examples.head(5)  # 5 worst fixtures
    good_fixtures = adjustment_examples.tail(5)  # 5 best fixtures
    
    print("\nPlayers with tough fixtures (showing raw vs adjusted predictions):")
    print(bad_fixtures.to_string(index=False))
    
    print("\nPlayers with favorable fixtures (showing raw vs adjusted predictions):")
    print(good_fixtures.to_string(index=False))
    
    # Print sample of adjustment impact
    sample_adjustments = gw31_data.sample(min(10, len(gw31_data)))
    print("\nSample of fixture adjustments:")
    print(sample_adjustments[['name', 'team', 'raw_predicted_goals', 'fixture_difficulty', 'fixture_adjusted_goals']].sort_values('fixture_difficulty'))
    
    # Blend model prediction with position baseline
    gw31_data['blended_prediction'] = (alpha * gw31_data['fixture_adjusted_goals']) + ((1-alpha) * gw31_data['position_avg_goals'])
    
    # Apply minimum probability floor, but maintain original predictions for higher values
    gw31_data['predicted_goals'] = gw31_data.apply(
        lambda row: max(row['blended_prediction'], row['min_probability']), axis=1
    )
    
    # Round to 2 decimal places for clean presentation
    gw31_data['predicted_goals'] = gw31_data['predicted_goals'].round(2)
    
    # Sort by predicted_goals in descending order
    gw31_data_sorted = gw31_data.sort_values(by="predicted_goals", ascending=False)
    
    # Calculate prediction statistics for analysis
    print("\nDistribution of predicted goal values:")
    prediction_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0]
    prediction_counts = pd.cut(gw31_data_sorted["predicted_goals"], bins=prediction_bins, include_lowest=True).value_counts().sort_index()
    print(prediction_counts)
    
    # Save the sorted predictions to a new file
    out_path = "GW31_Predicted_goals_with_fixtures.csv"
    columns_to_save = ["name", "team", "position", "GW", "opponent_name", "was_home", 
                      "fixture_difficulty", "predicted_goals", "raw_predicted_goals", 
                      "fixture_adjusted_goals", "blended_prediction", "min_probability",
                      # Keep the IDs for reference if needed
                      "opponent_team", "team_attacking", "opp_defending"]
    columns_available = [col for col in columns_to_save if col in gw31_data_sorted.columns]
    gw31_data_sorted[columns_available].to_csv(out_path, index=False)
    print(f"Sorted predictions for GW 31 saved to {out_path}")
    print("Sample predictions with fixture data:\n", 
          gw31_data_sorted[["name", "team", "position", "opponent_name", "was_home", "fixture_difficulty", "raw_predicted_goals", "fixture_adjusted_goals", "predicted_goals"]].head(10))
    
    # Save prediction distribution to file
    distribution_path = "GW31_Prediction_Distribution_with_fixtures.csv"
    prediction_counts.to_csv(distribution_path)
    print(f"Prediction distribution saved to {distribution_path}")
    
    # Optional: Create a histogram visualization of the prediction distribution
    try:
        plt.figure(figsize=(12, 6))
        plt.hist(gw31_data_sorted["predicted_goals"], bins=prediction_bins, alpha=0.7, color='royalblue')
        plt.title('Distribution of Predicted Goal Probabilities for GW31 (With Fixture Difficulty)')
        plt.xlabel('Predicted Goals')
        plt.ylabel('Number of Players')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('GW31_Goal_Distribution_with_fixtures.png')
        print("Distribution visualization saved to GW31_Goal_Distribution_with_fixtures.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    #  Create a visualization showing how fixture difficulty affects players
    try:
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of all players
        plt.scatter(gw31_data["fixture_difficulty"], gw31_data["raw_predicted_goals"], 
                   alpha=0.3, color='blue', label='Raw Predictions')
        plt.scatter(gw31_data["fixture_difficulty"], gw31_data["fixture_adjusted_goals"], 
                   alpha=0.3, color='red', label='Fixture-Adjusted Predictions')
        
        # Add trend lines
        x_range = np.linspace(gw31_data["fixture_difficulty"].min(), gw31_data["fixture_difficulty"].max(), 100)
        
        # Trend line for raw predictions
        z_raw = np.polyfit(gw31_data["fixture_difficulty"], gw31_data["raw_predicted_goals"], 1)
        p_raw = np.poly1d(z_raw)
        plt.plot(x_range, p_raw(x_range), '--', color='blue', 
                label=f'Raw Trend (slope: {z_raw[0]:.3f})')
        
        # Trend line for adjusted predictions
        z_adj = np.polyfit(gw31_data["fixture_difficulty"], gw31_data["fixture_adjusted_goals"], 1)
        p_adj = np.poly1d(z_adj)
        plt.plot(x_range, p_adj(x_range), '-', color='red', 
                label=f'Adjusted Trend (slope: {z_adj[0]:.3f})')
        
        # Calculate and display the fixture impact
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
        print("Fixture adjustment impact visualization saved to Fixture_Adjustment_Impact.png")
        
        # Create histogram comparing distributions
        plt.figure(figsize=(12, 6))
        plt.hist(gw31_data["raw_predicted_goals"], bins=20, alpha=0.5, color='blue', label='Raw Predictions')
        plt.hist(gw31_data["fixture_adjusted_goals"], bins=20, alpha=0.5, color='red', label='Fixture-Adjusted')
        plt.title('Distribution of Goal Predictions Before and After Fixture Adjustment')
        plt.xlabel('Predicted Goals')
        plt.ylabel('Number of Players')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Prediction_Distribution_Comparison.png')
        print("Prediction distribution comparison saved to Prediction_Distribution_Comparison.png")
        
    except Exception as e:
        print(f"Could not create fixture adjustment visualization: {e}")

if __name__ == "__main__":
    main()