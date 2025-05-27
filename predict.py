import pandas as pd
import joblib
from train_model import predict_upcoming_games  # Make sure this function is accessible

sfs = joblib.load("sfs_model.pkl")
predictors = joblib.load("predictors.pkl")

# Load latest preprocessed data (must match training format)
full = pd.read_csv("latest_full.csv")

# Run prediction
predictions = predict_upcoming_games(full, sfs.estimator_, predictors)
print(predictions)
