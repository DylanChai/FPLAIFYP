"""
Simple FPL Model Evaluation Script

This script provides a basic framework for evaluating different machine learning models
on FPL prediction tasks without requiring additional dependencies beyond scikit-learn and XGBoost.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor

def load_data(data_path):
    """Load the FPL data from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"Successfully loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

def evaluate_model(model, model_name, features, target, data, cutoff_gw, prediction_window=1):
    """
    Evaluate a model using data up to cutoff_gw to predict the next prediction_window gameweeks.
    """
    # Train on data up to cutoff_gw
    train_data = data[data["GW"] <= cutoff_gw]
    
    # Test on data from the next prediction_window gameweeks
    test_data = data[
        (data["GW"] > cutoff_gw) & 
        (data["GW"] <= cutoff_gw + prediction_window)
    ]
    
    if train_data.empty:
        raise ValueError(f"No training data found for GW <= {cutoff_gw}")
    
    if test_data.empty:
        raise ValueError(f"No test data found for GW {cutoff_gw+1} to {cutoff_gw+prediction_window}")
    
    # Prepare features and target
    for col in features + [target]:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    
    # Drop rows with missing values
    train_data = train_data.dropna(subset=features + [target])
    test_data = test_data.dropna(subset=features + [target])
    
    X_train = train_data[features]
    y_train = train_data[target]
    
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Create results dictionary
    results = {
        "model": model_name,
        "target": target,
        "cutoff_gw": cutoff_gw,
        "prediction_window": prediction_window,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
    
    return results

def compare_models(models_dict, features, target, data, evaluation_gws, prediction_windows=None):
    """
    Compare multiple models across different evaluation gameweeks.
    """
    if prediction_windows is None:
        prediction_windows = [1]
    
    results = []
    
    for model_name, model in models_dict.items():
        for gw in evaluation_gws:
            for window in prediction_windows:
                try:
                    result = evaluate_model(
                        model, model_name, features, target, data, gw, window
                    )
                    results.append(result)
                    print(f"Evaluated {model_name} at GW {gw}, window {window}: MAE = {result['mae']:.4f}")
                except Exception as e:
                    print(f"Error evaluating {model_name} for GW {gw}, window {window}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def visualize_results(results_df, metric='mae', by='model', save_path=None):
    """Visualize evaluation results."""
    plt.figure(figsize=(12, 8))
    
    if by == 'model':
        # Group by model
        pivot = results_df.pivot_table(
            values=metric, 
            index='cutoff_gw', 
            columns='model', 
            aggfunc='mean'
        )
        pivot.plot(marker='o', ax=plt.gca())
        plt.title(f'{metric.upper()} by Model and Gameweek')
        plt.xlabel('Cutoff Gameweek')
        plt.ylabel(metric.upper())
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "external", "Fantasy-Premier-League", 
                          "data", "2024-25", "gws", "merged_gw_cleaned.csv")
    
    # Create results directory
    results_dir = "simple_evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load data
    try:
        data = load_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Define models to evaluate
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42)
    }
    
    # Try to add XGBoost if available
    try:
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        print("XGBoost model added to evaluation.")
    except Exception as e:
        print(f"Skipping XGBoost model due to error: {e}")
    
    # Define features for goals prediction (example)
    goal_features = [
        "minutes", "threat", "creativity", "ict_index", "starts"
    ]
    
    # Add optional features if available
    optional_features = ["xP", "expected_goals", "expected_goal_involvements"]
    for feature in optional_features:
        if feature in data.columns:
            goal_features.append(feature)
    
    # Set evaluation parameters
    evaluation_gws = [10, 15, 20]  # Train on data up to these gameweeks
    prediction_windows = [1, 2]     # Predict this many gameweeks ahead
    
    # Run evaluation for goals prediction
    results = compare_models(
        models_dict=models,
        features=goal_features,
        target="goals_scored",
        data=data,
        evaluation_gws=evaluation_gws,
        prediction_windows=prediction_windows
    )
    
    # Save results
    results.to_csv(f"{results_dir}/goals_model_comparison.csv", index=False)
    
    # Visualize results
    visualize_results(
        results_df=results,
        metric='mae',
        by='model',
        save_path=f"{results_dir}/goals_model_comparison.png"
    )
    
    # Print best model
    best_model = results.groupby('model')['mae'].mean().idxmin()
    best_mae = results.groupby('model')['mae'].mean().min()
    print(f"\nBest model for goals prediction: {best_model} (MAE: {best_mae:.4f})")
    
    print(f"\nEvaluation complete. Results saved to '{results_dir}' directory")

if __name__ == "__main__":
    main()