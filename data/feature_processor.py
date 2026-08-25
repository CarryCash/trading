import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_validator import validate_data
from core.indicators import compute_all
from core.candlestick_patterns import detect_all

def process_features(df_clean=None):
    if df_clean is None:
        df_clean = validate_data()
        
    df_features = df_clean.copy()
    
    # 1. Aplicar indicadores
    indicators_df = compute_all(df_features)
    
    # 2. Aplicar patrones
    # detect_all() currently takes a df and returns a dict for the *last* row.
    # For historical processing, we need to apply it to rolling windows or use a vectorized version.
    # Since detect_all uses rolling logic in some places but is designed for a single row evaluation (.iloc[-1]),
    # we need to iterate or rewrite it. Let's do a fast apply.
    
    # We will import the individual functions from candlestick_patterns and apply them properly.
    import core.candlestick_patterns as cdl
    
    # Vectorized / rolling evaluations for patterns:
    def _body(df): return (df["close"] - df["open"]).abs()
    def _range(df): return (df["high"] - df["low"]).replace(0, 1e-9)
    def _upper_wick(df): return df["high"] - df[["open", "close"]].max(axis=1)
    def _lower_wick(df): return df[["open", "close"]].min(axis=1) - df["low"]
    
    body = _body(df_features)
    rng = _range(df_features)
    upper_wick = _upper_wick(df_features)
    lower_wick = _lower_wick(df_features)
    
    indicators_df["pattern_doji"] = (body <= 0.1 * rng).astype(int)
    indicators_df["pattern_hammer"] = ((lower_wick > 2 * body) & (upper_wick < body)).astype(int)
    indicators_df["pattern_shooting_star"] = ((upper_wick > 2 * body) & (lower_wick < body)).astype(int)
    
    # Engulfing
    prev_close = df_features["close"].shift(1)
    prev_open = df_features["open"].shift(1)
    
    bullish_engulfing = (
        (prev_close < prev_open) & 
        (df_features["close"] > df_features["open"]) & 
        (df_features["close"] >= prev_open) & 
        (df_features["open"] <= prev_close)
    )
    indicators_df["pattern_bullish_engulfing"] = bullish_engulfing.astype(int)
    
    bearish_engulfing = (
        (prev_close > prev_open) & 
        (df_features["close"] < df_features["open"]) & 
        (df_features["open"] >= prev_close) & 
        (df_features["close"] <= prev_open)
    )
    indicators_df["pattern_bearish_engulfing"] = bearish_engulfing.astype(int)
    
    # Combine engulfing into one pattern column as requested or separated
    indicators_df["pattern_engulfing"] = bullish_engulfing.astype(int) - bearish_engulfing.astype(int)
    
    # Three soldiers
    last3_close_up = (df_features["close"] > df_features["open"]) & (df_features["close"].shift(1) > df_features["open"].shift(1)) & (df_features["close"].shift(2) > df_features["open"].shift(2))
    last3_increasing = (df_features["close"] > df_features["close"].shift(1)) & (df_features["close"].shift(1) > df_features["close"].shift(2))
    indicators_df["pattern_three_soldiers"] = (last3_close_up & last3_increasing).astype(int)
    
    # 3. Generar labels (forward-looking 30 min = 2 velas)
    indicators_df['returns_2_candles'] = indicators_df['close'].shift(-2) / indicators_df['close'] - 1
    
    # Drop rows at the end that don't have future labels (last 2 rows)
    indicators_df = indicators_df.iloc[:-2].copy()
    
    # Rebalancear labels con cuantiles (Opción A)
    indicators_df['LABEL'] = pd.qcut(
        indicators_df['returns_2_candles'],
        q=3,
        labels=[-1, 0, 1],  # SELL, HOLD, BUY
        duplicates='drop'
    )
    
    print("\n=== REPORTE DE DISTRIBUCIÓN DE CLASES ===")
    print(indicators_df['LABEL'].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    print("=========================================\n")
    
    # 4. Guardar
    processed_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_file = os.path.join(processed_dir, 'features_balanced.csv')
    indicators_df.to_csv(out_file, index=False)
    
    print(f"Procesado exitosamente. Output guardado en {out_file}")
    return indicators_df

if __name__ == "__main__":
    process_features()
