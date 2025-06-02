import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import requests
import joblib
from scipy.special import expit


def add_target(group):
    group = group.copy()  # prevent fragmentation warning
    group["target"] = group["won"].shift(-1)
    return group


def predict_upcoming_games(data_pug, model, predictors_pug):

    train = data_pug[data_pug["target"].isin([0, 1])]
    test = data_pug[data_pug["target"] == 2].copy()

    model.fit(train[predictors_pug], train["target"])

    # Getting the decision scores and convert the score to a probability via sigmoid
    decision_scores = model.decision_function(test[predictors_pug])
    test["score"] = decision_scores
    test["win_probability"] = expit(decision_scores)

    test["prediction"] = model.predict(test[predictors_pug])

    test["prediction"] = test["prediction"].map({0: "loss", 1: "win"})

    test["game_id"] = test.apply(
        lambda row: "_".join(sorted([row["team_x"], row["team_y"]])) + "_" + str(row["date_next"]),
        axis=1
    )

    # We keep only the row (team perspective) with the higher confidence score
    test = test.sort_values("score", ascending=False).drop_duplicates("game_id")

    output = pd.DataFrame({
        "home_team": test["team_x"],
        "opponent": test["team_y"],
        "prediction": test["prediction"],
        "win_probability": test["win_probability"],
        "date": test["date_next"]
    })

    return output.reset_index(drop=True)


# def predict_upcoming_games(data_pug, model, predictors_pug):
#     # Split the data
#     train = data_pug[data_pug["target"].isin([0, 1])]
#     test = data_pug[data_pug["target"] == 2].copy()
#
#     # Train the model
#     model.fit(train[predictors_pug], train["target"])
#
#     # Predict
#     predictions = model.predict(test[predictors_pug])
#     predictions = pd.Series(predictions, index=test.index)
#
#     # Build the output DataFrame
#     output = pd.DataFrame(index=test.index)
#
#     # Rename and include relevant context columns
#     output["team"] = test["team_x"]
#     output["opponent"] = test["team_y"]
#     output["prediction"] = predictions
#     output["date"] = test["date_next"]
#
#     output.reset_index(drop=True, inplace=True)
#
#     return output


def shift_col(team_sc, col_name):
    next_col = team_sc[col_name].shift(-1)
    return next_col


def add_col(df_ac, col_name):
    return df_ac.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))


def merge_next_game(row, team_mappings, next_games, reverse_team_mappings):
    abbrev = row["team"]
    full_name = team_mappings.get(abbrev)
    game = next_games.get(full_name)

    # If the game is missing or any critical field is None, skip update
    if not game or game["home"] is None or game["visitor"] is None or game["date"] is None:
        return pd.Series([None, None, None])

    # Map full names to abbreviations
    home_abbrev = reverse_team_mappings.get(game["home"])
    visitor_abbrev = reverse_team_mappings.get(game["visitor"])
    date = game["date"]

    # Determine opponent
    if full_name == game["home"]:
        opponent_abbrev = visitor_abbrev
    elif full_name == game["visitor"]:
        opponent_abbrev = home_abbrev
    else:
        return pd.Series([None, None, None])

    return pd.Series([home_abbrev, opponent_abbrev, date])


# def find_team_averages(team):
#     rolling = team.rolling(10).mean()
#     return rolling
def find_team_averages(team_fta):
    numeric_cols = team_fta.select_dtypes(include='number')  # only numeric columns
    rolling = numeric_cols.rolling(10).mean()
    return rolling


def fill_from_incomplete(row, lookup):
    if pd.isna(row["home_next"]) and pd.isna(row["team_opp_next"]) and pd.isna(row["date_next"]):
        team = row["team"]
        if team in lookup.index:
            replacement = lookup.loc[team]
            return pd.Series([
                replacement["home_next"],
                replacement["team_opp_next"],
                replacement["date_next"]
            ])
    return pd.Series([row["home_next"], row["team_opp_next"], row["date_next"]])


if __name__ == "__main__":
    df = pd.read_csv("nba_games.csv", index_col=0)

    del df["mp.1"]
    del df["mp_opp.1"]
    del df["index_opp"]

    df = df.groupby("team", group_keys=False).apply(add_target)
    df = df.copy()
    # df["target"][pd.isnull(df["target"])] = 2
    df.loc[pd.isnull(df["target"]), "target"] = 2
    df["target"] = df["target"].astype(int, errors="ignore")

    nulls = pd.isnull(df).sum()
    nulls = nulls[nulls > 0]
    valid_columns = df.columns[~df.columns.isin(nulls.index)]

    df = df[valid_columns].copy()

    rr = RidgeClassifier(alpha=1)

    split = TimeSeriesSplit(n_splits=3)

    sfs = SequentialFeatureSelector(rr,
                                    n_features_to_select=30,
                                    direction="forward",
                                    cv=split,
                                    n_jobs=1
                                    )

    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    sfs.fit(df[selected_columns], df["target"])

    predictors = list(selected_columns[sfs.get_support()])

    df.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])

    df_rolling = df[list(selected_columns) + ["won", "team", "season"]]

    df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

    rolling_cols = [f"{col}_10" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols
    df = pd.concat([df, df_rolling], axis=1)

    df = df.dropna()

    df["home_next"] = add_col(df, "home")
    df["team_opp_next"] = add_col(df, "team_opp")
    df["date_next"] = add_col(df, "date")

    latest_df = df.copy()

    # Convert date_next to datetime to ensure proper sorting if needed

    latest_df["date_next"] = pd.to_datetime(latest_df["date_next"], errors="coerce")

    # Ensure team identifiers are available
    team_col = "team"

    # Sort the dataframe by date and select the latest game for each team
    latest_df_sorted = latest_df.sort_values("date_next", na_position='last')
    latest_games_per_team = latest_df_sorted.groupby(team_col).tail(1)

    incomplete = latest_games_per_team[
        latest_games_per_team["home_next"].isna() |
        latest_games_per_team["team_opp_next"].isna() |
        latest_games_per_team["date_next"].isna()
        ]

    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    data = response.json()

    # List upcoming games:
    games = data["leagueSchedule"]["gameDates"]
    upcoming_games = []

    from datetime import datetime

    today = datetime.today().date()

    for day in games:
        game_date = datetime.strptime(day["gameDate"], "%m/%d/%Y %H:%M:%S").date()

        if game_date >= today:
            for g in day["games"]:
                upcoming_games.append({
                    "date": game_date.isoformat(),
                    "visitor": g["awayTeam"]["teamName"],
                    "home": g["homeTeam"]["teamName"]
                })

    # Get next game for each team
    next_game_for_team = {}
    for g in sorted(upcoming_games, key=lambda x: x["date"]):
        for team in [g["home"], g["visitor"]]:
            if team not in next_game_for_team:
                next_game_for_team[team] = g

    # print(next_game_for_team)
    # print(len(next_game_for_team))

    # Mapping from 3-letter codes to full names
    bref_to_full = {
        'PHI': '76ers', 'BOS': 'Celtics', 'SAS': 'Spurs', 'WAS': 'Wizards', 'IND': 'Pacers',
        'SAC': 'Kings', 'POR': 'Trail Blazers', 'MIA': 'Heat', 'HOU': 'Rockets', 'OKC': 'Thunder',
        'DAL': 'Mavericks', 'MEM': 'Grizzlies', 'ORL': 'Magic', 'LAL': 'Lakers', 'CHI': 'Bulls',
        'ATL': 'Hawks', 'NYK': 'Knicks', 'BRK': 'Nets', 'MIN': 'Timberwolves', 'PHO': 'Suns',
        'LAC': 'Clippers', 'DEN': 'Nuggets', 'CHO': 'Hornets', 'GSW': 'Warriors', 'UTA': 'Jazz',
        'TOR': 'Raptors', 'CLE': 'Cavaliers', 'DET': 'Pistons', 'NOP': 'Pelicans', 'MIL': 'Bucks'
    }

    # Reverse it to map full names back to abbreviations
    full_to_bref = {v: k for k, v in bref_to_full.items()}

    # Apply and update the DataFrame
    incomplete[["home_next", "team_opp_next", "date_next"]] = incomplete.apply(
        lambda row: merge_next_game(row, bref_to_full, next_game_for_team, full_to_bref), axis=1
    )

    incomplete_lookup = incomplete.set_index("team")[["home_next", "team_opp_next", "date_next"]]

    # Apply to update only missing values
    latest_df[["home_next", "team_opp_next", "date_next"]] = latest_df.apply(
        lambda row: fill_from_incomplete(row, incomplete_lookup), axis=1
    )

    full = latest_df.merge(latest_df[rolling_cols + ["team_opp_next", "date_next", "team"]],
                           left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

    full["date_next"] = full["date_next"].astype(str)

    removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns

    selected_columns = full.columns[~full.columns.isin(removed_columns)]
    sfs.fit(full[selected_columns], full["target"])
    assert hasattr(sfs, "n_features_in_"), "SequentialFeatureSelector was not fitted properly."
    predictors = list(selected_columns[sfs.get_support()])

    # Saving the trained SequentialFeatureSelector with model inside
    joblib.dump(sfs, "sfs_model.pkl")
    # joblib.dump(sfs.estimator_, "model.pkl")

    # Saving the selected predictors used for future prediction
    joblib.dump(predictors, "predictors.pkl")

    full.to_csv("latest_full.csv")

    print("Model, predictors, and scaler saved successfully.")









