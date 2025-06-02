# NBA Game Outcome Predictor 🏀

This project predicts the outcome of upcoming NBA games using machine learning trained on historical game data from the 2016–2024 NBA seasons.


## ⚙️ Overview:

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

## 🧪 Example Output (of a prediction):

| home team | opponent | prediction | probability (of the home team winning) | date       |
|------|----------|------------|----------------------|------------|
| IND  | NYK      | 1 (win)    |   0.54444                   |2025-05-29 |

As such it displays the prediction for each of the 30 teams next game (so the table displays 15 rows if all teams have a next game to play within the current season). It only displays the prediction if the team has a next game so during the playoffs it only displays the appropriate predictions based on which teams are playing.

5. **Testing Accuracy**:
  `testing_accuracy.py`:
  - Tests the model's accuracy on historical data 

6. **Installing Dependencies**:
   - pip install -r requirements.txt
  

## ⚙️ How it Works:
1. Run get_data.py in order to obtain the historical game data required to train the RidgeClassifer model.
2. Run parse_data.py in order to transform the data within the 'data' directory into a pandas dataframe that can be used to train the RidgeClassifier model.
3. Run train_model.py in order to create and train the model using pandas dataframne created from running parse_data.py. Currently the 'nba_games.csv' has games from the beginning of the 2018 season to March 28, 2025, and the saved model (sfs_model.pkl) is trained on all games within this time period. If you would like to obtain data from the latest games and further train the model on such games as well, then run this file.
4. Run predict.py to obtain the predictions for each team's next game.
5. (Optional) Run testing_accuracy.py to test the accuracy of the model on historical data.
