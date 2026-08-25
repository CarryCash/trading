import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add core to path just in case, though not needed here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, "data", "processed")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

def engineer_features(df):
    df_eng = df.copy()
    
    # A) MOMENTUM Y CAMBIO DE PRECIO
    df_eng['price_momentum_1'] = df_eng['close'].pct_change(1) * 100
    df_eng['price_momentum_5'] = df_eng['close'].pct_change(5) * 100
    df_eng['price_momentum_20'] = df_eng['close'].pct_change(20) * 100
    
    # B) VOLATILIDAD RELATIVA
    df_eng['volatility_20_pct'] = (df_eng['close'].rolling(20).std() / df_eng['close']) * 100
    df_eng['atr_ratio'] = (df_eng['atr'] / df_eng['close']) * 100
    
    # C) INDICATOR CROSSOVERS Y CAMBIOS
    df_eng['sma_cross'] = (df_eng['sma_50'] > df_eng['sma_200']).astype(int)
    df_eng['ema_above_sma'] = (df_eng['ema_20'] > df_eng['sma_50']).astype(int)
    df_eng['macd_positive'] = (df_eng['macd'] > 0).astype(int)
    df_eng['price_above_200sma'] = (df_eng['close'] > df_eng['sma_200']).astype(int)
    
    # D) FUERZAS EXTREMAS (Oversold/Overbought)
    df_eng['rsi_oversold'] = (df_eng['rsi'] < 30).astype(int)
    df_eng['rsi_overbought'] = (df_eng['rsi'] > 70).astype(int)
    df_eng['stoch_oversold'] = (df_eng['stoch_k'] < 20).astype(int)
    
    # E) VOLUMEN Y CONFIRMACIÓN
    vol_sma_20 = df_eng['volume'].rolling(20).mean()
    df_eng['volume_spike_ratio'] = (df_eng['volume'] / vol_sma_20).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_eng['obv_trending'] = (df_eng['obv'] > df_eng['obv'].shift(1)).astype(int)
    
    # F) SOPORTE/RESISTENCIA (Distancia y Posición)
    supp = df_eng['support'].replace(0, np.nan)
    res = df_eng['resistance'].replace(0, np.nan)
    
    df_eng['distance_to_support_pct'] = ((df_eng['close'] - supp) / supp) * 100
    df_eng['distance_to_resistance_pct'] = ((res - df_eng['close']) / res) * 100
    
    upper_third_threshold = supp + (2/3) * (res - supp)
    df_eng['price_in_upper_third'] = (df_eng['close'] > upper_third_threshold).astype(int)
    
    # Additional useful derivations for absolute price indicators
    if 'vwap' in df_eng.columns:
        df_eng['vwap_distance_pct'] = ((df_eng['close'] - df_eng['vwap']) / df_eng['vwap']) * 100
        
    if 'bb_lower' in df_eng.columns and 'bb_upper' in df_eng.columns:
        bb_range = (df_eng['bb_upper'] - df_eng['bb_lower']).replace(0, np.nan)
        df_eng['bb_position_pct'] = ((df_eng['close'] - df_eng['bb_lower']) / bb_range) * 100
        
    return df_eng

def process():
    filepath = os.path.join(processed_dir, 'features_balanced.csv')
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    
    # Engineer Features
    df_eng = engineer_features(df)
    
    # Features engineered CSV with all data (including NaN warmup rows for completeness in output 1)
    out_csv = os.path.join(processed_dir, 'features_engineered.csv')
    df_eng.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(df_eng)} rows and {df_eng.shape[1]} columns.")
    
    # To drop absolute prices and non-predictive variables:
    cols_to_drop = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'sma_50', 'sma_200', 'ema_20', 'support', 'resistance',
        'vwap', 'bb_mid', 'bb_upper', 'bb_lower', # Absolute prices
        'returns_2_candles', 'LABEL'
    ]
    
    features_cols = [c for c in df_eng.columns if c not in cols_to_drop]
    
    # Drop rows with NaN (which will be at least the first 200 candles due to SMA200 and momentum_20)
    # Using subset to ensure we only drop if NaNs are in our predictive features or labels
    df_clean = df_eng.dropna(subset=features_cols + ['LABEL']).reset_index(drop=True)
    
    X = df_clean[features_cols]
    y = df_clean['LABEL'].values
    
    # Train / Val / Test Split (80% / 10% / 10%)
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    X_train = X.iloc[:train_end].copy()
    y_train = y[:train_end]
    
    X_val = X.iloc[train_end:val_end].copy()
    y_val = y[train_end:val_end]
    
    X_test = X.iloc[val_end:].copy()
    y_test = y[val_end:]
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(processed_dir, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    
    with open(os.path.join(processed_dir, 'feature_names.json'), 'w') as f:
        json.dump(features_cols, f, indent=4)
        
    # Normalization Report
    report_features = []
    for i, col in enumerate(features_cols):
        # determine if binary
        unique_vals = set(X_train[col].unique())
        data_type = "binary" if unique_vals.issubset({0, 1}) else "continuous"
        
        report_features.append({
            "name": col,
            "mean_train": float(scaler.mean_[i]),
            "std_train": float(scaler.scale_[i]),
            "min_train": float(X_train[col].min()),
            "max_train": float(X_train[col].max()),
            "data_type": data_type
        })
        
    report = {
        "features": report_features,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "normalization_method": "StandardScaler (z-score)"
    }
    
    with open(os.path.join(processed_dir, 'normalization_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Artifacts saved successfully.")
    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_val shape: {X_val_scaled.shape}")
    print(f"X_test shape: {X_test_scaled.shape}")
    
    # Validate properties
    assert not np.isnan(X_train_scaled).any(), "NaNs found in X_train_scaled"
    assert not np.isinf(X_train_scaled).any(), "Infs found in X_train_scaled"
    
    # Distribution check on train
    y_train_series = pd.Series(y_train)
    dist = y_train_series.value_counts(normalize=True).mul(100).round(2)
    print(f"\ny_train distribution:\n{dist}")

if __name__ == "__main__":
    process()
