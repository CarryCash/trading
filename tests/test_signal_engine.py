import pytest
import pandas as pd
from strategy.signal_engine import _vote_trend, _vote_momentum, evaluate_mtf, SignalResult

def test_vote_trend():
    row = pd.Series({
        "sma_50": 100,
        "sma_200": 90,  # 50 > 200 -> Alcista
        "close": 105,
        "ema_20": 100,  # close > ema20 -> Alcista
        "macd_hist": 0.5 # hist > 0 -> Alcista
    })
    votes = _vote_trend(row)
    assert votes["sma_cross"] == 1
    assert votes["ema_trend"] == 1
    assert votes["macd"] == 1

def test_vote_momentum():
    row = pd.Series({
        "rsi": 25,       # < 30 -> Alcista
        "stoch_k": 85,   # > 80 -> Bajista
        "cci": -150      # < -100 -> Alcista
    })
    votes = _vote_momentum(row)
    assert votes["rsi"] == 1
    assert votes["stochastic"] == -1
    assert votes["cci"] == 1

def test_evaluate_mtf_missing_primary():
    dfs = {"4h": pd.DataFrame()}
    res = evaluate_mtf(dfs, "15m")
    assert res.action == "HOLD"
    assert "Falta timeframe principal" in res.reasons[0]

def test_evaluate_mtf_empty_primary():
    dfs = {"15m": pd.DataFrame()}
    res = evaluate_mtf(dfs, "15m")
    assert res.action == "HOLD"
    assert "DataFrame principal vacío" in res.reasons[0]
