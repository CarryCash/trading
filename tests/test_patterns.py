import pytest
import pandas as pd
from core import candlestick_patterns as cdl

def test_is_hammer():
    # Vela con cuerpo pequeño arriba, mecha inferior larga
    df = pd.DataFrame([{
        "open": 100,
        "high": 101.5,
        "low": 80,
        "close": 101
    }])
    assert cdl.is_hammer(df) == True
    
def test_not_hammer():
    # Cuerpo muy grande
    df = pd.DataFrame([{
        "open": 100,
        "high": 105,
        "low": 80,
        "close": 90
    }])
    assert cdl.is_hammer(df) == False

def test_bullish_engulfing():
    # Vela 1: Roja, Vela 2: Verde que envuelve
    df = pd.DataFrame([
        {"open": 100, "high": 105, "low": 90, "close": 95},  # Roja
        {"open": 90, "high": 110, "low": 85, "close": 105}   # Verde, envuelve la anterior
    ])
    assert cdl.is_bullish_engulfing(df) == True

def test_bearish_engulfing():
    # Vela 1: Verde, Vela 2: Roja que envuelve
    df = pd.DataFrame([
        {"open": 95, "high": 105, "low": 90, "close": 100},  # Verde
        {"open": 105, "high": 110, "low": 85, "close": 90}   # Roja, envuelve
    ])
    assert cdl.is_bearish_engulfing(df) == True

def test_doji():
    # Apertura y cierre casi idénticos
    df = pd.DataFrame([{
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 100.2
    }])
    assert cdl.is_doji(df) == True
