# Fantasy Premier League Prediction Dashboard  
**Final Year Project | BSc Computer Science**

A suite of machine-learning models and a Streamlit dashboard to help Fantasy Premier League managers make data-driven decisions: predicting goals, assists, clean sheets, cards, bonus points—and optimising a full £100 m squad.

---

## 📂 Repository Structure

/ ├── external/vaastav_fpl/ # Raw data from vaastav/Fantasy-Premier-League (unmodified) ├── models/ # All model scripts, outputs & dashboard │ ├── train_goalsTeamStrength.py │ ├── train_assists.py │ ├── train_cards.py │ ├── train_bonus.py │ ├── train_cleansheets.py │ ├── optimise.py │ ├── dashboard.py │ └── GW<nn>Predicted*.csv # Model outputs ├── data/ # (Optional) local CSV snapshots used during development └── README.md # This file

yaml
Copy code

- **`external/vaastav_fpl/`**  
  Contains `merged_gw_cleaned.csv`, `teams.csv`, `fixtures.csv` from the original dataset (no modifications).

- **`models/`**  
  Core feature engineering, model training, evaluation logic, and the Streamlit dashboard. Outputs like `GW34_Predicted_goals.csv` will be saved here.

- **`data/`**  
  directory for storing cleaned Merged data

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DylanChai/FPLAIFYP.git
   cd fpl-prediction
Create and activate a virtual environment

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
Install dependencies

bash
Copy code
pip install -r models/requirements.txt
Verify data availability
Ensure external/vaastav_fpl/ contains:

merged_gw_cleaned.csv

teams.csv

fixtures.csv

▶️ Usage
1. Train & Predict
Run model scripts to generate predictions for the next Gameweek:

bash
Copy code
python models/train_goalsTeamStrength.py     # Predict goals
python models/train_assists.py               # Predict assists
python models/train_cards.py                 # Predict cards
python models/train_bonus.py                 # Predict bonus points
python models/train_cleansheets.py           # Predict clean sheets
Each script outputs a file:

php-template
Copy code
models/GW<nn>_Predicted_<model>.csv
Example: GW34_Predicted_goals.csv

2. Launch the Dashboard
Start the dashboard to visualise predictions and build an optimised team:

bash
Copy code
streamlit run models/dashboard.py
Dashboard Features:

Overview – Legend and FPL scoring explanation

Tabs – Goals, Assists, Clean Sheets, Cards, Bonus

Visuals – Fixture difficulty scatter plots, Top-25 tables

Total Points – Aggregated model output for expected points

Optimiser – Build a 15-player squad under budget/position constraints

Load My Team – Enter your FPL team ID to import your squad

⚽ Optimiser
Formulated as a mixed-integer linear program using PuLP:

£100 million budget

15 players: 2 GK, 5 DEF, 5 MID, 3 FWD

Max 3 players per club

Objective: Maximise total predicted FPL points

📖 Acknowledgements
vaastav/Fantasy-Premier-League — data source

Libraries: scikit-learn, pandas, NumPy, Streamlit, PuLP

Community resources & AI-assisted prototyping

- **`References`**  
Vaastav. (2023). Fantasy-Premier-League: Data for Fantasy Premier League. GitHub. https://github.com/vaastav/Fantasy-Premier-League

https://medium.com/%40joseph.m.oconnor.88/linearly-optimising-fantasy-premier-league-teams-3b76e9694877 | MILP 

All feature engineering, model development, integration & optimisation logic authored and validated by me

