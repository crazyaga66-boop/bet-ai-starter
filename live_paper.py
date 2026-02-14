# live_paper.py
import os
import json
import pandas as pd
import numpy as np
from datetime import date
from xgboost import XGBClassifier

from config import SafetyConfig
from features import build_elo_sequential, add_common_features, add_soccer_features, add_hockey_features
from safety import DailyRiskState, kill_switch_active, enforce_allowlist, within_odds_limits, daily_limits_ok

def load_model(path: str) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model

def suggest_bets_soccer(df_live, df_hist, injuries, sc: SafetyConfig, bankroll: float, state: DailyRiskState):
    model = load_model("out/models/soccer_xgb.json")

    # Build features using history + live rows (Elo needs sequence)
    df_all = pd.concat([df_hist, df_live], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.strftime("%Y-%m-%d")
    df_all = build_elo_sequential(df_all, allow_draws=True)

    # Split back
    live_rows = df_all.tail(len(df_live)).copy()
    live_rows = add_common_features(live_rows, injuries)
    live_rows = add_soccer_features(live_rows)

    feats = ["home_imp", "draw_imp", "away_imp", "elo_diff", "injury_diff", "rest_diff"]
    probs = model.predict_proba(live_rows[feats])

    suggestions = []
    for i, r in live_rows.reset_index(drop=True).iterrows():
        if kill_switch_active(sc.KILL_SWITCH_PATH):
            print("KILL SWITCH ACTIVE — stopping suggestions.")
            break

        if not enforce_allowlist(r["home_team"], r["away_team"], sc.ALLOWED_TEAMS):
            continue

        edges = {
            "home": probs[i][2] - r["home_imp"],
            "draw": probs[i][1] - r["draw_imp"],
            "away": probs[i][0] - r["away_imp"],
        }
        pick = max(edges, key=edges.get)
        edge = edges[pick]

        odds = r[f"{pick}_odds"]
        if edge < sc.EDGE_THRESHOLD:
            continue
        if not within_odds_limits(odds, sc.MIN_ODDS, sc.MAX_ODDS):
            continue
        if not daily_limits_ok(state, bankroll, sc.MAX_BETS_PER_DAY, sc.DAILY_LOSS_LIMIT_PCT):
            continue

        stake = bankroll * sc.MAX_BET_PCT
        suggestions.append({
            "sport": "soccer",
            "date": r["date"],
            "match": f'{r["home_team"]} vs {r["away_team"]}',
            "pick": pick,
            "odds": float(odds),
            "edge": float(edge),
            "suggested_stake": round(stake, 2)
        })

        state.bets_placed += 1

    return suggestions

def suggest_bets_hockey(df_live, df_hist, injuries, goalies, sc: SafetyConfig, bankroll: float, state: DailyRiskState):
    model = load_model("out/models/hockey_xgb.json")

    df_all = pd.concat([df_hist, df_live], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.strftime("%Y-%m-%d")
    df_all = build_elo_sequential(df_all, allow_draws=False)

    live_rows = df_all.tail(len(df_live)).copy()
    live_rows = add_common_features(live_rows, injuries)
    live_rows = add_hockey_features(live_rows, goalies)

    feats = ["home_imp", "away_imp", "elo_diff", "injury_diff", "rest_diff", "goalie_diff"]
    probs = model.predict_proba(live_rows[feats])

    suggestions = []
    for i, r in live_rows.reset_index(drop=True).iterrows():
        if kill_switch_active(sc.KILL_SWITCH_PATH):
            print("KILL SWITCH ACTIVE — stopping suggestions.")
            break

        if not enforce_allowlist(r["home_team"], r["away_team"], sc.ALLOWED_TEAMS):
            continue

        edges = {
            "home": probs[i][1] - r["home_imp"],
            "away": probs[i][0] - r["away_imp"],
        }
        pick = max(edges, key=edges.get)
        edge = edges[pick]

        odds = r[f"{pick}_odds"]
        if edge < sc.EDGE_THRESHOLD:
            continue
        if not within_odds_limits(odds, sc.MIN_ODDS, sc.MAX_ODDS):
            continue
        if not daily_limits_ok(state, bankroll, sc.MAX_BETS_PER_DAY, sc.DAILY_LOSS_LIMIT_PCT):
            continue

        stake = bankroll * sc.MAX_BET_PCT
        suggestions.append({
            "sport": "hockey",
            "date": r["date"],
            "match": f'{r["home_team"]} vs {r["away_team"]}',
            "pick": pick,
            "odds": float(odds),
            "edge": float(edge),
            "suggested_stake": round(stake, 2)
        })

        state.bets_placed += 1

    return suggestions

def main():
    sc = SafetyConfig()

    bankroll = sc.BANKROLL_START
    state = DailyRiskState(day=date.today(), start_bankroll=bankroll)

    soccer_hist = pd.read_csv("data/soccer_matches.csv")
    soccer_live = pd.read_csv("data/live_odds_soccer.csv")
    soccer_inj = pd.read_csv("data/soccer_injuries.csv") if os.path.exists("data/soccer_injuries.csv") else pd.DataFrame()

    hockey_hist = pd.read_csv("data/hockey_matches.csv")
    hockey_live = pd.read_csv("data/live_odds_hockey.csv")
    hockey_inj = pd.read_csv("data/hockey_injuries.csv") if os.path.exists("data/hockey_injuries.csv") else pd.DataFrame()
    hockey_goalies = pd.read_csv("data/hockey_goalies.csv") if os.path.exists("data/hockey_goalies.csv") else pd.DataFrame()

    soccer_suggestions = suggest_bets_soccer(soccer_live, soccer_hist, soccer_inj, sc, bankroll, state)
    hockey_suggestions = suggest_bets_hockey(hockey_live, hockey_hist, hockey_inj, hockey_goalies, sc, bankroll, state)

    print("\n=== SAFE BET SUGGESTIONS (ALERTS ONLY) ===")
    for s in soccer_suggestions + hockey_suggestions:
        print(s)

if __name__ == "__main__":
    main()
