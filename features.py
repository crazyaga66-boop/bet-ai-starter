# features.py
import numpy as np
import pandas as pd

def compute_injury_impact(inj_df, match_date, team):
    if inj_df is None or inj_df.empty:
        return 0.0
    rows = inj_df[(inj_df["date"] == match_date) & (inj_df["team"] == team) & (inj_df["is_out"] == 1)]
    return float(rows["impact"].sum()) if len(rows) else 0.0

def compute_goalie_rating(goalie_df, match_date, team):
    if goalie_df is None or goalie_df.empty:
        return 0.0
    rows = goalie_df[(goalie_df["date"] == match_date) & (goalie_df["team"] == team) & (goalie_df["is_confirmed"] == 1)]
    return float(rows["goalie_rating"].iloc[0]) if len(rows) else 0.0

def build_elo_sequential(df, k=20.0, allow_draws=True):
    df = df.sort_values("date").reset_index(drop=True)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    elo = {t: 1500.0 for t in teams}

    home_elo, away_elo = [], []

    def expected(a, b):
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        home_elo.append(elo.get(h, 1500.0))
        away_elo.append(elo.get(a, 1500.0))

        if "home_goals" in r and "away_goals" in r and not pd.isna(r["home_goals"]):
            if r["home_goals"] > r["away_goals"]:
                score_h = 1.0
            elif r["home_goals"] < r["away_goals"]:
                score_h = 0.0
            else:
                score_h = 0.5 if allow_draws else 0.0

            eh = expected(elo[h], elo[a])
            elo[h] = elo[h] + k * (score_h - eh)
            elo[a] = elo[a] + k * ((1.0 - score_h) - (1.0 - eh))

    df["elo_diff"] = np.array(home_elo) - np.array(away_elo)
    return df

def add_common_features(df, inj_df):
    df = df.copy()
    df["home_imp"] = 1.0 / df["home_odds"]
    df["away_imp"] = 1.0 / df["away_odds"]
    df["inj_home"] = df.apply(lambda r: compute_injury_impact(inj_df, r["date"], r["home_team"]), axis=1)
    df["inj_away"] = df.apply(lambda r: compute_injury_impact(inj_df, r["date"], r["away_team"]), axis=1)
    df["injury_diff"] = df["inj_home"] - df["inj_away"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    return df

def add_soccer_features(df):
    df = df.copy()
    df["draw_imp"] = 1.0 / df["draw_odds"]
    return df

def add_hockey_features(df, goalie_df):
    df = df.copy()
    df["home_goalie_rating"] = df.apply(lambda r: compute_goalie_rating(goalie_df, r["date"], r["home_team"]), axis=1)
    df["away_goalie_rating"] = df.apply(lambda r: compute_goalie_rating(goalie_df, r["date"], r["away_team"]), axis=1)
    df["goalie_diff"] = df["home_goalie_rating"] - df["away_goalie_rating"]
    return df
