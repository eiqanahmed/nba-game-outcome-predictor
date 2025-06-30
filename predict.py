import pandas as pd
import joblib
import requests
from datetime import datetime
from scipy.special import expit


def predict_upcoming_games(data_pug, model_pug, predictors_pug):

    train = data_pug[data_pug["target"].isin([0, 1])]
    test = data_pug[data_pug["target"] == 2].copy()

    if test.empty:
        print("No upcoming games available for prediction.")
        return

    model_pug.fit(train[predictors_pug], train["target"])

    # Getting the decision scores and convert the score to a probability via sigmoid
    decision_scores = model_pug.decision_function(test[predictors_pug])
    test["score"] = decision_scores
    test["win_probability"] = expit(decision_scores)

    test["prediction"] = model_pug.predict(test[predictors_pug])

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


def update_with_next_games(latest_df):
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    data = response.json()

    games = data["leagueSchedule"]["gameDates"]
    today = datetime.today().date()

    upcoming_games = []
    for day in games:
        game_date = datetime.strptime(day["gameDate"], "%m/%d/%Y %H:%M:%S").date()
        if game_date >= today:
            for g in day["games"]:
                upcoming_games.append({
                    "date": game_date.isoformat(),
                    "visitor": g["awayTeam"]["teamName"],
                    "home": g["homeTeam"]["teamName"]
                })

    bref_to_full = {
        'PHI': '76ers', 'BOS': 'Celtics', 'SAS': 'Spurs', 'WAS': 'Wizards', 'IND': 'Pacers',
        'SAC': 'Kings', 'POR': 'Trail Blazers', 'MIA': 'Heat', 'HOU': 'Rockets', 'OKC': 'Thunder',
        'DAL': 'Mavericks', 'MEM': 'Grizzlies', 'ORL': 'Magic', 'LAL': 'Lakers', 'CHI': 'Bulls',
        'ATL': 'Hawks', 'NYK': 'Knicks', 'BRK': 'Nets', 'MIN': 'Timberwolves', 'PHO': 'Suns',
        'LAC': 'Clippers', 'DEN': 'Nuggets', 'CHO': 'Hornets', 'GSW': 'Warriors', 'UTA': 'Jazz',
        'TOR': 'Raptors', 'CLE': 'Cavaliers', 'DET': 'Pistons', 'NOP': 'Pelicans', 'MIL': 'Bucks'
    }
    full_to_bref = {v: k for k, v in bref_to_full.items()}

    next_game_for_team = {}
    for g in sorted(upcoming_games, key=lambda x: x["date"]):
        for team in [g["home"], g["visitor"]]:
            if team not in next_game_for_team:
                next_game_for_team[team] = g

    def merge_next_game(row):
        abbrev = row["team"]
        full_name = bref_to_full.get(abbrev)
        game = next_game_for_team.get(full_name)
        if not game or game["home"] is None or game["visitor"] is None or game["date"] is None:
            return pd.Series([None, None, None])
        home_abbrev = full_to_bref.get(game["home"])
        visitor_abbrev = full_to_bref.get(game["visitor"])
        date = game["date"]
        opponent_abbrev = visitor_abbrev if full_name == game["home"] else home_abbrev
        return pd.Series([home_abbrev, opponent_abbrev, date])

    latest_df[["home_next", "team_opp_next", "date_next"]] = latest_df.apply(merge_next_game, axis=1)
    return latest_df


if __name__ == "__main__":
    sfs = joblib.load("sfs_model.pkl")
    # model = joblib.load("model.pkl")
    predictors = joblib.load("predictors.pkl")

    full = pd.read_csv("latest_full.csv")
    full["team"] = full["team_x"]
    full = update_with_next_games(full)
    full = full.dropna(subset=["date_next"])
    model = sfs.estimator

    predictions = predict_upcoming_games(full, model, predictors)
    # predictions = predict_upcoming_games(full, model, predictors)
    print(predictions)
