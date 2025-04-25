import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Step 1: Set up file path as in your RF model
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(
    base_dir, 
    "..", 
    "external", 
    "Fantasy-Premier-League", 
    "data", 
    "2024-25", 
    "gws", 
    "merged_gw_cleaned.csv"
)

# Step 2: Load your data.
data = pd.read_csv(data_path, low_memory=False)
print("Data loaded with shape:", data.shape)

# Step 3: Define your features and target.
features = ["minutes", "threat", "creativity", "ict_index", "starts"]
target = "goals_scored"

# Step 4: Convert features and target to numeric.
for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Step 5: Drop rows with missing values in the specified columns.
data = data.dropna(subset=features + [target])
print("Data shape after dropping missing values:", data.shape)

# Step 6: Split the data into training and test sets.
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize the XGBoost model with some basic parameters.
xgb_model = XGBRegressor(
    n_estimators=100,      # number of trees
    learning_rate=0.1,     # shrinkage to prevent overfitting
    max_depth=3,           # maximum depth of each tree
    random_state=42
)

# Step 8: Train the model on the training data.
xgb_model.fit(X_train, y_train)

# Step 9: Make predictions on the test set.
predictions = xgb_model.predict(X_test)

# Step 10: Evaluate model performance using Mean Absolute Error.
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (XGBoost):", mae)
