import pandas as pd
from sklearn.metrics import accuracy_score
import joblib


def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []

    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)


if __name__ == "__main__":
    sfs = joblib.load("sfs_model.pkl")
    # model = joblib.load("model.pkl")
    predictors = joblib.load("predictors.pkl")

    full = pd.read_csv("latest_full.csv")

    model = sfs.estimator

    predictions = backtest(full, model, predictors)

    accuracy = accuracy_score(predictions["actual"], predictions["prediction"])

    print(accuracy)
