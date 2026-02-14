# train_backtest.py
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from config import SafetyConfig, ModelConfig
from features import build_elo_sequential, add_common_features, add_soccer_features, add_hockey_features

def ensure_dirs():
    os.makedirs("out/models", exist_ok=True)
    os.makedirs("out/logs", exist_ok=True)

def soccer_target(df):
    return np.where(df["home_goals"] > df["away_goals"], 2,
           np.where(df["home_goals"] == df["away_goals"], 1, 0))

def hockey_target(df):
    return np.where(df["home_goals"] > df["away_goals"], 1, 0)

def walk_forward(df):
    seasons = sorted(df["season"].unique())
    for i in range(1, len(seasons)):
        yield seasons[:i], [seasons[i]]

def train_soccer():
    sc, mc = SafetyConfig(), ModelConfig()
    df = pd.read_csv("data/soccer_matches.csv")
    inj = pd.read_csv("data/soccer_injuries.csv") if os.path.exists("data/soccer_injuries.csv") else pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = build_elo_sequential(df, allow_draws=True)
    df = add_common_features(df, inj)
    df = add_soccer_features(df)
    df["result"] = soccer_target(df)

    feats = ["home_imp", "draw_imp", "away_imp", "elo_diff", "injury_diff", "rest_diff"]

    reports = []
    for train_seasons, test_season in walk_forward(df):
        tr = df[df["season"].isin(train_seasons)]
        te = df[df["season"].isin(test_season)]
        model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, eval_metric="mlogloss")
        model.fit(tr[feats], tr["result"])
        probs = model.predict_proba(te[feats])
        ll = log_loss(te["result"], probs, labels=[0,1,2])
        reports.append({"train_seasons": train_seasons, "test_season": test_season[0], "logloss": ll})

    ensure_dirs()
    pd.DataFrame(reports).to_csv("out/logs/soccer_report.csv", index=False)
    model.save_model("out/models/soccer_xgb.json")
    print("Soccer trained & report saved.")

def train_hockey():
    df = pd.read_csv("data/hockey_matches.csv")
    inj = pd.read_csv("data/hockey_injuries.csv") if os.path.exists("data/hockey_injuries.csv") else pd.DataFrame()
    goalies = pd.read_csv("data/hockey_goalies.csv") if os.path.exists("data/hockey_goalies.csv") else pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = build_elo_sequential(df, allow_draws=False)
    df = add_common_features(df, inj)
    df = add_hockey_features(df, goalies)
    df["result"] = hockey_target(df)

    feats = ["home_imp", "away_imp", "elo_diff", "injury_diff", "rest_diff", "goalie_diff"]

    model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, eval_metric="logloss")
    model.fit(df[feats], df["result"])

    ensure_dirs()
    model.save_model("out/models/hockey_xgb.json")
    print("Hockey trained.")

if __name__ == "__main__":
    train_soccer()
    train_hockey()
