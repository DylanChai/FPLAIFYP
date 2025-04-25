import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt

def main():
    # 1. Load the merged file with updated path
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

    print("Columns in merged_gw_cleaned.csv:", data.columns.tolist())
    print("Sample rows:\n", data.head())

    # 2. Update training data to use GW 1-30
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 30)]

    # Check if training data is available
    if train_data.empty:
        print("No training data found for GW 1-30 in merged_gw_cleaned.csv.")
        return

    print(f"Training data shape (GW 1-30): {train_data.shape}")

    # 3. Check if position data is available
    has_position_data = 'position' in train_data.columns
    print(f"Position data available: {has_position_data}")
    
    # 4. Create position-specific features if position data is available
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

    # 5. Select features & target - REMOVED BPS
    features = [
        "minutes",           # How long the player played
        "xP",                # Expected points (given in CSV)
        "threat",            # Attacking threat
        # "bps",             # REMOVED Bonus points system
        "transfers_in",      # Possibly a proxy for form
        "expected_goals",    # Direct goal expectation
        "creativity",        # Added for creativity metric
        "ict_index",         # Added for overall influence
        "starts",            # Added to capture regular starters
        "is_forward",        # Position features
        "is_midfielder",
        "is_defender",
        "is_goalkeeper"
    ]

    # Check if all features exist
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        # Remove missing features from the list
        features = [f for f in features if f in train_data.columns]
        print(f"Using these features instead: {features}")

    target = "goals_scored"

    # 6. Convert columns to numeric if needed
    for col in features + [target]:
        if col in train_data.columns:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 7. Create a copy of training data with complete target values for model training
    train_data_with_target = train_data.dropna(subset=[target])
    
    # 8. Calculate average goals by position for baseline predictions
    position_goals = {}
    if has_position_data:
        position_goals = train_data_with_target.groupby('position')[target].mean().to_dict()
        print("Average goals by position:", position_goals)
    else:
        # Default values if position data not available
        position_goals = {'FWD': 0.15, 'MID': 0.08, 'DEF': 0.03, 'GK': 0.001, 'Unknown': 0.05}
    
    # 9. Use imputation instead of dropping rows with missing feature values
    imputer = SimpleImputer(strategy='mean')
    
    # Fit the imputer on the training data
    X_train_with_target = train_data_with_target[features].copy()
    
    # Handle any remaining NaN values for imputation
    for col in X_train_with_target.columns:
        if X_train_with_target[col].isna().any():
            X_train_with_target[col] = X_train_with_target[col].fillna(0)
    
    imputer.fit(X_train_with_target)
    
    # Transform the feature data for training
    X_train_imputed = imputer.transform(X_train_with_target)
    
    # Convert back to DataFrame for clarity
    X_train_df = pd.DataFrame(X_train_imputed, columns=features, index=train_data_with_target.index)
    y_train = train_data_with_target[target]

    # 10. Split the imputed training data into train/test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train, test_size=0.2, random_state=42)

    # 11. Train a RandomForestRegressor model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=200,           # More trees for better probability distribution
        min_samples_leaf=5,         # Increase to prevent overfitting/extreme predictions
        max_features='sqrt',        # Standard practice for random forests
        random_state=42,
        n_jobs=-1                   # Use all available cores
    )
    model.fit(X_train, y_train)

    # 12. Evaluate the model on the test set
    predictions_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print("Mean Absolute Error (MAE) on the test set:", mae)

    # 13. Feature importance analysis
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # 14. Get all unique players from the dataset for prediction
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

    # 15. Prepare player stats for GW31 prediction by averaging their previous performances
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    player_avg_stats = train_data.groupby(["name", "team"])[numeric_columns].mean().reset_index()
    
    # 16. Ensure ALL players are included for prediction, even those with limited data
    gw31_data = all_players.merge(player_avg_stats, on=["name", "team"], how="left")
    gw31_data["GW"] = 31  # Set to GW 31
    
    # 17. Create position-specific features for prediction
    # Handle position-based features
    gw31_data['is_forward'] = (gw31_data['position'] == 'FWD').astype(int)
    gw31_data['is_midfielder'] = (gw31_data['position'] == 'MID').astype(int) 
    gw31_data['is_defender'] = (gw31_data['position'] == 'DEF').astype(int)
    gw31_data['is_goalkeeper'] = (gw31_data['position'] == 'GK').astype(int)
    
    # 18. Impute any missing values for prediction
    X_future = gw31_data[features].copy()
    
    # Handle any remaining NaN values before imputation
    for col in X_future.columns:
        if X_future[col].isna().any():
            X_future[col] = X_future[col].fillna(0)
    
    X_future_imputed = imputer.transform(X_future)
    X_future_df = pd.DataFrame(X_future_imputed, columns=features)
    
    # 19. Get raw model predictions
    raw_predictions = model.predict(X_future_df)
    
    # 20. Function to get position-based minimum probability - MORE AGGRESSIVE
    def get_min_probability(row):
        position = row.get('position', 'Unknown')
        minutes_avg = row.get('minutes', 0)
        if pd.isna(minutes_avg):
            minutes_avg = 0
            
        # Base minimum probability by position - INCREASED VALUES
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
    
    # 21. Apply a more realistic smoothing to prediction values
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
    
    # Blend model prediction with position baseline
    gw31_data['blended_prediction'] = (alpha * gw31_data['raw_predicted_goals']) + ((1-alpha) * gw31_data['position_avg_goals'])
    
    # Apply minimum probability floor, but maintain original predictions for higher values
    gw31_data['predicted_goals'] = gw31_data.apply(
        lambda row: max(row['blended_prediction'], row['min_probability']), axis=1
    )
    
    # 22. Round to 2 decimal places for clean presentation
    gw31_data['predicted_goals'] = gw31_data['predicted_goals'].round(2)
    
    # 23. Sort by predicted_goals in descending order
    gw31_data_sorted = gw31_data.sort_values(by="predicted_goals", ascending=False)
    
    # 24. Calculate prediction statistics for analysis
    print("\nDistribution of predicted goal values:")
    prediction_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0]
    prediction_counts = pd.cut(gw31_data_sorted["predicted_goals"], bins=prediction_bins, include_lowest=True).value_counts().sort_index()
    print(prediction_counts)
    
    # 25. Save the sorted predictions to a new file
    out_path = "GW31_Predicted_goals_no_bps.csv"
    columns_to_save = ["name", "team", "position", "GW", "predicted_goals", "minutes", "raw_predicted_goals", "blended_prediction", "min_probability"]
    columns_available = [col for col in columns_to_save if col in gw31_data_sorted.columns]
    gw31_data_sorted[columns_available].to_csv(out_path, index=False)
    print(f"Sorted predictions for GW 31 saved to {out_path}")
    print("Sample predictions:\n", gw31_data_sorted[["name", "team", "GW", "predicted_goals"]].head(10))
    
    # 26. Save prediction distribution to file
    distribution_path = "GW31_Prediction_Distribution_no_bps.csv"
    prediction_counts.to_csv(distribution_path)
    print(f"Prediction distribution saved to {distribution_path}")
    
    # 27. Optional: Create a histogram visualization of the prediction distribution
    try:
        plt.figure(figsize=(12, 6))
        plt.hist(gw31_data_sorted["predicted_goals"], bins=prediction_bins, alpha=0.7, color='royalblue')
        plt.title('Distribution of Predicted Goal Probabilities for GW31 (No BPS)')
        plt.xlabel('Predicted Goals')
        plt.ylabel('Number of Players')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('GW31_Goal_Distribution_no_bps.png')
        print("Distribution visualization saved to GW31_Goal_Distribution_no_bps.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")

if __name__ == "__main__":
    main()