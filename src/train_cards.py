import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("Starting cards prediction model...")

    # 1. Load the merged gameweek file using the cleaned CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "external", "Fantasy-Premier-League", "data", "2024-25", "gws", "merged_gw_cleaned.csv")
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        print("Please run the CSV cleaner script first to create the cleaned CSV file.")
        return

    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"Successfully loaded merged_gw_cleaned.csv with shape: {data.shape}")
    except Exception as e:
        print(f"Error reading merged_gw_cleaned.csv: {e}")
        return

    # 2. Update to use gameweeks 1-30 as training data
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 30)]
    if train_data.empty:
        print("No training data found for GW 1-30 in merged_gw_cleaned.csv.")
        return
    print(f"Training data shape (GW 1-30): {train_data.shape}")

    # 3. Convert columns to numeric
    for col in train_data.columns:
        if col not in ["name", "team", "position", "kickoff_time"]:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 4. Check for card columns
    print("Available columns:", train_data.columns.tolist())
    
    card_columns = [col for col in train_data.columns if "card" in col.lower()]
    print(f"Found card-related columns: {card_columns}")
    
    if "yellow_cards" in train_data.columns and "red_cards" in train_data.columns:
        train_data["total_cards"] = train_data["yellow_cards"].fillna(0) + train_data["red_cards"].fillna(0)
        target_column = "total_cards"
    elif "yellow_cards" in train_data.columns:
        target_column = "yellow_cards"
        train_data[target_column] = train_data[target_column].fillna(0)
    elif "red_cards" in train_data.columns:
        target_column = "red_cards"
        train_data[target_column] = train_data[target_column].fillna(0)
    else:
        print("No card data found. Exiting.")
        return
    
    print(f"Using {target_column} as the target for prediction")

    # 5. Select features for card prediction
    card_features = [
        "minutes",           # Playing time
        "total_points",      # Overall performance
        "bps",               # Bonus points system
        "influence",         # Player's influence on the game
    ]
    
    # Add threat if available
    if "threat" in train_data.columns:
        card_features.append("threat")
    
    # Add was_home if available
    if "was_home" in train_data.columns:
        card_features.append("was_home")
    
    # Get position information if available
    if "position" in train_data.columns:
        train_data = pd.get_dummies(train_data, columns=["position"])
        position_cols = [col for col in train_data.columns if col.startswith("position_")]
        card_features.extend(position_cols)
        print("Position column found and one-hot encoded.")
    else:
        print("Position column not found. Continuing without position encoding.")
    
    # Check if all features exist in the dataset
    missing_features = [f for f in card_features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        # Remove missing features from the list
        card_features = [f for f in card_features if f in train_data.columns]
    
    print(f"Using these features for card prediction: {card_features}")

    # 6. Prepare data for modeling
    train_data = train_data.dropna(subset=[target_column])
    
    # Handle missing values in features by filling with zeros
    X_cards = train_data[card_features].fillna(0)
    y_cards = train_data[target_column]
    
    print(f"Data shape after preparation: X={X_cards.shape}, y={y_cards.shape}")

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_cards, y_cards, test_size=0.2, random_state=42)
    
    # 8. Train the model (using RandomForestRegressor for cards)
    model_cards = RandomForestRegressor(n_estimators=100, random_state=42)
    model_cards.fit(X_train, y_train)
    
    # 9. Evaluate the model
    predictions_test = model_cards.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    
    print(f"Model evaluation:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # 10. Prepare GW 31 data for prediction (UPDATED)
    # Group by player and team, averaging all features
    gw31_data_cards = train_data.groupby(["name", "team"])[card_features].mean().reset_index()
    gw31_data_cards["GW"] = 31  # Updated to GW 31
    
    # Use the full feature set for prediction
    X_future_cards = gw31_data_cards[card_features].fillna(0)
    
    # 11. Predict cards for GW 31
    gw31_data_cards["predicted_cards"] = model_cards.predict(X_future_cards)
    
    # Ensure predictions are non-negative
    gw31_data_cards["predicted_cards"] = np.clip(gw31_data_cards["predicted_cards"], 0, None)
    
    # 12. Sort the predictions by predicted_cards in descending order
    gw31_data_cards_sorted = gw31_data_cards.sort_values(by="predicted_cards", ascending=False)
    
    # 13. Save the sorted predictions to a CSV file
    out_path = "GW31_Predicted_Cards.csv"  # Updated filename
    gw31_data_cards_sorted.to_csv(out_path, index=False)
    
    print(f"Sorted predictions for GW 31 saved to {out_path}")
    print("Top 10 players most likely to receive cards:")
    print(gw31_data_cards_sorted[["name", "team", "GW", "predicted_cards"]].head(10))
    
    # 14. Feature importance
    feature_importance = pd.DataFrame({
        'Feature': card_features,
        'Importance': model_cards.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance for Card Prediction:")
    print(feature_importance)

if __name__ == "__main__":
    main()