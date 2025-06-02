# NBA Game Outcome Predictor 🏀

This project predicts the outcome of upcoming NBA games using machine learning trained on historical NBA game data.


## ⚙️ Overview:

1. **Data Collection**:  
   `get_data.py` scrapes NBA games from [basketball-reference.com](https://www.basketball-reference.com) using Playwright. Note: Currently the 'data/standings' directory in this repository has games from the beginning of the 2018 season to March 28, 2025.

2. **Data Parsing**:  
   `parse_data.py` converts raw HTML box scores into a pandas dataframe that is saved as a CSV file. Running this file gives you a CSV file titled "nba_games.csv" which contains game data from the last 7 seasons. Note: The repository already contains a "nba_games.csv" file which has data from the beginning of the 2018 season to March 28, 2025.

3. **Model Training**:  
   `train_model.py`:
   - Cleans the dataset (`nba_games.csv`)
   - Adds rolling averages, feature selection, and target columns
   - Trains a Ridge Classifier using `SequentialFeatureSelector`
   - Saves:
     - `sfs_model.pkl` – the trained selector with model
     - `predictors.pkl` – the chosen feature columns
     - `latest_full.csv` - this file will give you the final dataset that is used for prediction.
   - Note: `latest_full.csv.zip` is in this repository, unzipping this file will give you the final dataset that is used for prediction. If you would like to obtain a new version of this file that contains data from the latest nba games, run `train_model.py` after you have run `get_data.py`, followed by `parse_data.py`.

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
1. Run get_data.py in order to obtain the historical game data required to train the RidgeClassifer model. Currently the 'data/scores' has games from the beginning of the 2018 season to March 28, 2025. If you would like to update this directory with data from the latest games, then run get_data.py. 
2. Run parse_data.py in order to transform the data within the 'data' directory into a pandas dataframe that can be used to train the RidgeClassifier model. Currently the 'nba_games.csv' has games from the beginning of the 2018 season to March 28, 2025. If you would like an upated version of this file (include data from the latest games) on then run parse_data.py.
3. Run train_model.py in order to create and train the model using pandas dataframe created from running parse_data.py. Currently the the saved model (sfs_model.pkl) is trained on all games from the beginning of the 2018 season to March 28, 2025. If you have rerun get_data.py followed by parse_data.py, then you can run train_model.py to train your model on the latest games as well.
4. Run predict.py to obtain the predictions for each team's next game.
5. (Optional) Run testing_accuracy.py to test the accuracy of the model on historical data.
