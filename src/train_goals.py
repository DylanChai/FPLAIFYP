import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import os

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

    # 3. Select features & target
    features = [
        "minutes",       # How long the player played
        "xP",            # Expected points (given in your CSV)
        "threat",        # Attacking threat
        "bps",           # Bonus points system
        "transfers_in",  # Possibly a proxy for form
        "expected_goals" # Another good feature if your merged_gw.csv includes it
    ]

    # Check if all features exist
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        # Remove missing features from the list
        features = [f for f in features if f in train_data.columns]
        print(f"Using these features instead: {features}")

    target = "goals_scored"

    # 4. Convert columns to numeric if needed
    for col in features + [target]:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 5. Create a copy of training data with complete target values for model training
    train_data_with_target = train_data.dropna(subset=[target])
    
    # 6. Use imputation instead of dropping rows with missing feature values
    # This is a key change to ensure we don't lose players due to missing data
    imputer = SimpleImputer(strategy='mean')
    
    # Fit the imputer on the training data
    imputer.fit(train_data_with_target[features].fillna(0))
    
    # Transform the feature data for training
    X_train_imputed = imputer.transform(train_data_with_target[features].fillna(0))
    
    # Convert back to DataFrame for clarity
    X_train_df = pd.DataFrame(X_train_imputed, columns=features, index=train_data_with_target.index)
    y_train = train_data_with_target[target]

    # 7. Split the imputed training data into train/test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train, test_size=0.2, random_state=42)

    # 8. Train a RandomForestRegressor model with more trees and minimum samples
    # Adjust min_samples_leaf to smooth predictions and avoid exactly zero values
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,  # Helps prevent exact zero predictions
        random_state=42
    )
    model.fit(X_train, y_train)

    # 9. Evaluate the model on the test set
    predictions_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print("Mean Absolute Error (MAE) on the test set:", mae)

    # 10. Get all unique players from the dataset for prediction
    all_players = data[["name", "team"]].drop_duplicates()
    print(f"Total unique players in dataset: {len(all_players)}")

    # 11. Prepare player stats for GW31 prediction by averaging their previous performances
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    player_avg_stats = train_data.groupby(["name", "team"])[numeric_columns].mean().reset_index()
    
    # 12. Ensure ALL players are included for prediction, even those with limited data
    gw31_data = all_players.merge(player_avg_stats, on=["name", "team"], how="left")
    gw31_data["GW"] = 31  # Set to GW 31
    
    # 13. Impute any missing values for prediction
    # Create a DataFrame with just the features needed for prediction
    X_future = gw31_data[features].fillna(0)
    X_future_imputed = imputer.transform(X_future)
    X_future_df = pd.DataFrame(X_future_imputed, columns=features)
    
    # 14. Predict goals for GW 31 with minimum floor value for all players
    raw_predictions = model.predict(X_future_df)
    
    # Add a small minimum probability to all predictions (0.001 = 0.1% chance)
    # This ensures no player has exactly 0% chance of scoring
    min_probability = 0.001
    gw31_data["predicted_goals"] = np.maximum(raw_predictions, min_probability)
    
    # 15. Sort by predicted_goals in descending order
    gw31_data_sorted = gw31_data.sort_values(by="predicted_goals", ascending=False)
    
    # 16. Calculate prediction statistics for analysis
    total_predictions = len(gw31_data_sorted)
    non_zero_predictions = len(gw31_data_sorted[gw31_data_sorted["predicted_goals"] > min_probability])
    zero_predictions = total_predictions - non_zero_predictions
    
    print(f"Total players with predictions: {total_predictions}")
    print(f"Players with above-minimum predictions: {non_zero_predictions}")
    print(f"Players with minimum probability: {zero_predictions}")
    
    # 17. Save the sorted predictions to a new file
    out_path = "GW31_Predicted_goals.csv"
    gw31_data_sorted.to_csv(out_path, index=False)
    print(f"Sorted predictions for GW 31 saved to {out_path}")
    print("Sample predictions:\n", gw31_data_sorted[["name", "team", "GW", "predicted_goals"]].head(10))
    
    # 18. Save a histogram of prediction values for analysis
    prediction_counts = pd.cut(gw31_data_sorted["predicted_goals"], 
                               bins=[0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0], 
                               include_lowest=True).value_counts().sort_index()
    print("\nDistribution of predicted goal values:")
    print(prediction_counts)
    
    # 19. Save prediction distribution to file
    distribution_path = "GW31_Prediction_Distribution.csv"
    prediction_counts.to_csv(distribution_path)
    print(f"Prediction distribution saved to {distribution_path}")

if __name__ == "__main__":
    main()