import pandas as pd
import joblib
from train_model import predict_upcoming_games  # Make sure this function is accessible

sfs = joblib.load("sfs_model.pkl")
# model = joblib.load("model.pkl")
predictors = joblib.load("predictors.pkl")

full = pd.read_csv("latest_full.csv")

model = sfs.estimator

predictions = predict_upcoming_games(full, model, predictors)

print(predictions)
