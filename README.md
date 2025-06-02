# NBA Game Outcome Predictor 🏀

This project predicts the outcome of upcoming NBA games using machine learning trained on historical game data from the 2016–2024 NBA seasons.


## ⚙️ How It Works

1. **Data Collection**:  
   `get_data.py` scrapes NBA games from [basketball-reference.com](https://www.basketball-reference.com) using Playwright.

2. **Data Parsing**:  
   `parse_data.py` converts raw HTML box scores into structured CSV data.

3. **Model Training**:  
   `train_model.py`:
   - Cleans the dataset (`nba_games.csv`)
   - Adds rolling averages, feature selection, and target columns
   - Trains a Ridge Classifier using `SequentialFeatureSelector`
   - Saves:
     - `sfs_model.pkl` – the trained selector with model
     - `predictors.pkl` – the chosen feature columns
     - `latest_full.csv` – the final dataset used for prediction

4. **Prediction**:  
   `predict.py`:
   - Loads saved model and predictors
   - Predicts results for upcoming games
   - Displays each team’s expected outcome and matchup

5. **Testing Accuracy**:
  `testing_accuracy.py`:
  - Tests the model's accuracy on historical data 

## 🧪 Example Output (of a prediction):

| team | opponent | prediction | date       |
|------|----------|------------|------------|
| IND  | NYK      | 1 (win)    | 2025-05-29 |
| NYK  | IND      | 0 (loss)   | 2025-05-29 |


