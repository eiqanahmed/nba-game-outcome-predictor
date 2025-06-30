import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import joblib


def add_target(group):
    group = group.copy()  # prevent fragmentation warning
    group["target"] = group["won"].shift(-1)
    return group


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
        team_ffi = row["team"]
        if team_ffi in lookup.index:
            replacement = lookup.loc[team_ffi]
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
                                    direction="backward",
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

    df = df.copy()

    full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]],
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









