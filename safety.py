# safety.py
import os
from dataclasses import dataclass
from datetime import date as dt_date

@dataclass
class DailyRiskState:
    day: dt_date
    bets_placed: int = 0
    pnl: float = 0.0
    start_bankroll: float = 0.0

def kill_switch_active(path: str) -> bool:
    return os.path.exists(path)

def enforce_allowlist(team_a: str, team_b: str, allowed_teams: tuple[str, ...]) -> bool:
    if not allowed_teams:
        return True
    return (team_a in allowed_teams) and (team_b in allowed_teams)

def within_odds_limits(odds: float, min_odds: float, max_odds: float) -> bool:
    return (odds >= min_odds) and (odds <= max_odds)

def daily_limits_ok(state: DailyRiskState, bankroll: float, max_bets: int, daily_loss_limit_pct: float) -> bool:
    if state.bets_placed >= max_bets:
        return False
    if state.start_bankroll <= 0:
        return True
    loss = state.start_bankroll - bankroll
    return (loss / state.start_bankroll) < daily_loss_limit_pct
