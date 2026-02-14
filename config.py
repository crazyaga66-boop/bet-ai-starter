# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SafetyConfig:
    ALERTS_ONLY: bool = True
    BANKROLL_START: float = 1000.0
    MAX_BET_PCT: float = 0.01
    DAILY_LOSS_LIMIT_PCT: float = 0.05
    MAX_BETS_PER_DAY: int = 10
    EDGE_THRESHOLD: float = 0.04
    MIN_ODDS: float = 1.20
    MAX_ODDS: float = 12.0
    ALLOW_SOCCER_1X2: bool = True
    ALLOW_HOCKEY_ML: bool = True
    ALLOWED_TEAMS: tuple[str, ...] = tuple()
    KILL_SWITCH_PATH: str = "out/KILL_SWITCH"

@dataclass(frozen=True)
class ModelConfig:
    RANDOM_SEED: int = 42
    N_ESTIMATORS: int = 400
    MAX_DEPTH: int = 5
    LEARNING_RATE: float = 0.05
    SUBSAMPLE: float = 0.9
    COLSAMPLE_BYTREE: float = 0.9
