import pytest
import pandas as pd
import numpy as np

from core import indicators

def get_dummy_data(size=300):
    np.random.seed(42)
    # Generate random walk
    close = 100 + np.cumsum(np.random.randn(size))
    high = close + np.random.rand(size)
    low = close - np.random.rand(size)
    close_s = pd.Series(close)
    open_p = close_s.shift(1).fillna(100)
    vol = np.random.randint(1000, 5000, size)
    
    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol
    })

def test_sma():
    df = get_dummy_data(100)
    sma_val = indicators.sma(df, period=20)
    assert len(sma_val) == 100
    assert pd.isna(sma_val.iloc[0])
    assert pd.notna(sma_val.iloc[-1])

def test_macd():
    df = get_dummy_data(100)
    macd_df = indicators.macd(df)
    assert "macd" in macd_df.columns
    assert "signal" in macd_df.columns
    assert "hist" in macd_df.columns
    assert pd.notna(macd_df["macd"].iloc[-1])

def test_cci():
    df = get_dummy_data(50)
    cci_val = indicators.cci(df, period=20)
    assert pd.notna(cci_val.iloc[-1])
    # CCI is generally between -300 and 300
    assert -500 < cci_val.iloc[-1] < 500

def test_support_resistance_no_future_leakage():
    df = get_dummy_data(50)
    # En versiones anteriores usaba center=True lo que daba NaN al final
    sr = indicators.support_resistance(df, window=10)
    assert pd.notna(sr["support"].iloc[-1]), "Data leakage fix failed, support is NaN at the end"
    assert pd.notna(sr["resistance"].iloc[-1]), "Data leakage fix failed, resistance is NaN at the end"

def test_compute_all():
    df = get_dummy_data(250)
    out = indicators.compute_all(df)
    # Check that basic columns exist
    assert "sma_200" in out.columns
    assert "rsi" in out.columns
    # With 250 rows, the last row should have a valid sma_200
    assert pd.notna(out["sma_200"].iloc[-1])
