"""
predict.py

Updates latest_full.csv with each team's next game info:
  - home_next      (home team tricode for that next game)
  - team_opp_next  (opponent tricode)
  - date_next      (YYYY-MM-DD)

Then runs predictions using the saved feature selector + predictors.

IMPORTANT:
- This script is robust to environments with no DNS/internet (e.g., Kaggle).
- If internet fetch fails, it will try to read a local cached file:
    scheduleLeagueV2.json
  Place that file in the same directory as 'predict.py'.
"""

import os
import json
import requests
import joblib
import pandas as pd

from datetime import datetime
from scipy.special import expit

# -----------------------------
# Config
# -----------------------------
SCHEDULE_CACHE = "scheduleLeagueV2.json"
# local cache / uploaded file
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
# Basketball-Reference -> NBA API tricodes
BREF_TO_NBA = {
    "BRK": "BKN",
    "PHO": "PHX",
    "CHO": "CHA",
}

# NBA API -> Basketball-Reference tricodes
NBA_TO_BREF = {v: k for k, v in BREF_TO_NBA.items()}


def _parse_game_dt_et(day_obj: dict, game_obj: dict) -> datetime | None:
    """
    Return timezone-aware game datetime in America/New_York.
    Prefer UTC timestamps, then convert to ET.
    """

    for key in ("gameDateTimeUTC", "gameDateUTC", "gameDateTimeGMT"):
        s = game_obj.get(key)
        if not s:
            continue
        try:
            ss = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ss)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(ET)
        except Exception:
            pass

    # Fall back to Est fields if present
    for key in ("gameDateTimeEst", "gameDateEst"):
        s = game_obj.get(key)
        if not s:
            continue
        try:
            ss = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ss)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ET)
            else:
                dt = dt.astimezone(ET)
            return dt
        except Exception:
            pass

    # Last resort: day bucket (midnight ET)
    bucket = day_obj.get("gameDate")
    if bucket:
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(bucket, fmt).replace(tzinfo=ET)
                return dt
            except Exception:
                continue

    return None


# -----------------------------
# Predictions
# -----------------------------

def ensure_numeric_matrix(df, cols, scaler=None, fit=False):
    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)

    if scaler is not None:
        if fit:
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

        X = pd.DataFrame(X, columns=cols, index=df.index)

    return X, cols


def predict_upcoming_games(data_pug, model_pug, predictors_pug):
    # Train
    train = data_pug[data_pug["target"].isin([0, 1])].copy()
    if "team" not in train.columns and "team_x" in train.columns:
        train["team"] = train["team_x"]

    df = data_pug.copy()
    if "team" not in df.columns and "team_x" in df.columns:
        df["team"] = df["team_x"]

    # use a stable sort key for "latest"
    date_col = "date"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        snapshots = df.sort_values(date_col).groupby("team", as_index=False).tail(1).copy()
    else:
        snapshots = df.groupby("team", as_index=False).tail(1).copy()

    snapshots = snapshots.dropna(subset=["date_next", "team_opp_next", "home_next"]).copy()

    # Keep home-team rows only (one per game)
    test = snapshots[snapshots["team"] == snapshots["home_next"]].copy()

    # print("snapshot teams:", snapshots["team"].nunique())  # should be 30
    # print("home-team rows:", test.shape[0])  # should be 15
    if test.shape[0] != 15:
        print("missing home teams:", sorted(set(snapshots["home_next"]) - set(test["team"])))

    # Prepare output format
    test["team_x"] = test["home_next"]
    test["team_y"] = test["team_opp_next"]

    X_train, predictors_pug = ensure_numeric_matrix(
        train, predictors_pug, scaler=scaler, fit=False)

    y_train = train["target"].astype(int)
    model_pug.fit(X_train, y_train)

    X_test, _ = ensure_numeric_matrix(
        test, predictors_pug, scaler=scaler, fit=False)
    scores = model_pug.decision_function(X_test)
    test["win_probability"] = expit(scores)

    preds = model_pug.predict(X_test)
    test["prediction"] = pd.Series(preds, index=test.index).map({0: "loss", 1: "win"})

    output = pd.DataFrame({
        "home_team": test["team_x"],
        "opponent": test["team_y"],
        "prediction": test["prediction"],
        "win_probability": test["win_probability"],
        "date": test["date_next"],
    }).sort_values(["date", "home_team"]).reset_index(drop=True)

    output["home_team"] = output["home_team"].replace(BREF_TO_NBA)
    output["opponent"] = output["opponent"].replace(BREF_TO_NBA)

    return output


# -----------------------------
# Schedule fetching (offline-safe)
# -----------------------------
def fetch_schedule(local_path: str = SCHEDULE_CACHE):
    """
    Tries:
      1) local JSON file (offline safe)
      2) live fetch from cdn.nba.com (may fail)
    """
    # 1) Local first
    if os.path.exists(local_path):
        print(f"[info] using local schedule file: {local_path}")
        with open(local_path, "r") as f:
            return json.load(f)

    # 2) Live fetch
    try:
        r = requests.get(SCHEDULE_URL, headers=UA_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()

        # cache for later offline runs
        with open(local_path, "w") as f:
            json.dump(data, f)

        print(f"[info] fetched + cached schedule to: {local_path}")
        return data

    except Exception as e:
        print(f"[warn] live schedule fetch failed: {e}")
        print("[warn] no cached schedule available; cannot update next-game fields")
        return None


def _parse_game_date_iso(day_obj: dict, game_obj: dict) -> str | None:
    """
    Return YYYY-MM-DD for the game in America/New_York time.
    Prefer UTC timestamps and convert to ET (avoids off-by-one).
    """

    # 1) Prefer UTC-like fields and convert to ET
    for key in ("gameDateTimeUTC", "gameDateUTC", "gameDateTimeGMT"):
        s = game_obj.get(key)
        if not s:
            continue
        try:
            ss = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ss)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            dt = dt.astimezone(ET)
            return dt.date().isoformat()
        except Exception:
            pass

    # 2) If only "Est" fields exist, parse as ET
    for key in ("gameDateTimeEst", "gameDateEst"):
        s = game_obj.get(key)
        if not s:
            continue
        try:
            # If it's just a date "YYYY-MM-DD", return it directly
            if isinstance(s, str) and len(s) == 10 and s[4] == "-" and s[7] == "-":
                return s

            ss = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ss)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ET)
            else:
                dt = dt.astimezone(ET)
            return dt.date().isoformat()
        except Exception:
            pass

    # 3) Last resort: day bucket
    bucket = day_obj.get("gameDate")
    if bucket:
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(bucket, fmt).replace(tzinfo=ET)
                return dt.date().isoformat()
            except Exception:
                continue

    return None


def build_next_game_map(schedule_json: dict) -> dict:
    league = (schedule_json or {}).get("leagueSchedule", {})
    game_dates = league.get("gameDates", [])

    now_et = datetime.now(ET)

    upcoming_games = []
    for day in game_dates:
        for g in day.get("games", []):
            home = (g.get("homeTeam") or {}).get("teamTricode")
            away = (g.get("awayTeam") or {}).get("teamTricode")
            if not home or not away:
                continue

            # Common: gameStatus 1=scheduled, 2=live, 3=final (varies, but this is typical)
            status_num = g.get("gameStatus")
            status_txt = (g.get("gameStatusText") or "").lower()
            if status_num == 3 or "final" in status_txt:
                continue

            # --- Time filter: must be after now in ET ---
            dt_et = _parse_game_dt_et(day, g)
            if dt_et is None:
                continue

            # If tipoff already happened, skip, since we are predicting each team's next game
            if dt_et <= now_et:
                continue

            upcoming_games.append({
                "dt_et": dt_et,
                "date": dt_et.date().isoformat(),
                "home": home,
                "visitor": away
            })

    # Earliest upcoming game per team
    next_for_team = {}
    for game in sorted(upcoming_games, key=lambda x: x["dt_et"]):
        for team in (game["home"], game["visitor"]):
            if team not in next_for_team:
                next_for_team[team] = game

    out = {}
    for team, game in next_for_team.items():
        home = game["home"]
        away = game["visitor"]
        opp = away if team == home else home
        out[team] = {
            "date": game["date"],  # YYYY-MM-DD
            "home": home,
            "opp": opp,
        }

    return out


# -----------------------------
# Update latest_full with next-game info
# -----------------------------
def update_with_next_games(latest_df: pd.DataFrame, schedule_json: dict) -> pd.DataFrame:
    if schedule_json is None:
        return latest_df

    next_map = build_next_game_map(schedule_json)
    if not next_map:
        print("[warn] schedule parsed but produced no upcoming games")
        return latest_df

    out = latest_df.copy()

    if "team" not in out.columns:
        if "team_x" in out.columns:
            out["team"] = out["team_x"]
        else:
            raise ValueError("Dataframe must contain 'team' or 'team_x'")

    # Convert dataset team codes (bref) to NBA codes for schedule lookup
    out["team_nba"] = out["team"].replace(BREF_TO_NBA)

    def _lookup(team_nba: str):
        g = next_map.get(team_nba)
        if not g:
            return pd.Series([None, None, None])

        # Map schedule tricodes back to bref for consistency with your dataset/output
        home_bref = NBA_TO_BREF.get(g["home"], g["home"])
        opp_bref = NBA_TO_BREF.get(g["opp"], g["opp"])
        return pd.Series([home_bref, opp_bref, g["date"]])

    out[["home_next", "team_opp_next", "date_next"]] = out["team_nba"].apply(_lookup)
    return out


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    sfs = joblib.load("/kaggle/working/sfs_model.pkl")
    predictors = joblib.load("/kaggle/working/predictors.pkl")
    scaler = joblib.load("/kaggle/working/scaler.pkl")

    # Remove any predictors that are clearly non-numeric identifiers
    ban = {"team", "team_x", "team_y", "team_opp_next", "home_next", "date_next", "date"}
    predictors = [p for p in predictors if p not in ban]

    full = pd.read_csv("latest_full.csv")

    # Align team column for next-game lookup
    if "team" not in full.columns and "team_x" in full.columns:
        full["team"] = full["team_x"]

    # Fetch schedule
    schedule = fetch_schedule()

    # Update next-game columns (home_next/team_opp_next/date_next)
    full = update_with_next_games(full, schedule)
    print(full[["team", "team_opp_next", "home_next", "date_next"]].dropna().head(20))

    model = sfs.estimator

    predictions = predict_upcoming_games(full, model, predictors)
    print(predictions)
