import os
import sys
import pandas as pd
import numpy as np

# Ensure we can import from core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def validate_data():
    dfs = []
    # 1. Leer todos los CSVs
    raw_dir = os.path.join(os.path.dirname(__file__), "raw")
    for mes in [4, 5, 6, 7]:
        filepath = os.path.join(raw_dir, f"BTCUSDT-15m-2026-{mes:02d}.csv")
        if os.path.exists(filepath):
            # No header in Binance raw CSVs
            cols = ['timestamp_us', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore']
            df = pd.read_csv(filepath, names=cols)
            dfs.append(df)
        else:
            print(f"Warning: {filepath} not found.")
            
    if not dfs:
        raise ValueError("No CSV files found in data/raw/")
        
    # 2. Combinar y ordenar
    df_combined = pd.concat(dfs, ignore_index=True)
    # Parse timestamp (in microseconds since it has 16 digits usually, or whatever it is, pd.to_datetime handles unit)
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp_us'], unit='us')
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    
    # 3. Validaciones
    # Gaps
    expected_diff = pd.Timedelta(minutes=15)
    diffs = df_combined['timestamp'].diff()
    gaps = (diffs > expected_diff).sum()
    
    # Duplicates
    duplicates = df_combined.duplicated(subset=['timestamp']).sum()
    if duplicates > 0:
        df_combined = df_combined.drop_duplicates(subset=['timestamp'])
        
    # Invalid values
    invalid = ((df_combined['close'] < 0) | (df_combined['volume'] < 0) | (df_combined['high'] < df_combined['low'])).sum()
    
    # Outliers (>20% in one candle)
    pct_change = df_combined['close'].pct_change().abs()
    outliers = (pct_change > 0.20).sum()
    
    total_rows = len(df_combined)
    
    # 4. Generar reporte
    print("=== DATA VALIDATION REPORT ===")
    print(f"[{df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}] Total velas cargadas: {total_rows}")
    print(f"[{df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}] Fecha rango: {df_combined['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"[{df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}] Gaps encontrados: {gaps} [OK]" if gaps == 0 else f"Gaps encontrados: {gaps} [WARN]")
    print(f"[{df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}] Duplicados: {duplicates} [OK]" if duplicates == 0 else f"Duplicados: {duplicates} [WARN]")
    print(f"[{df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}] Valores inválidos: {invalid} [OK]" if invalid == 0 else f"Valores inválidos: {invalid} [WARN]")
    print(f"[{df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}] Outliers (>20%): {outliers} (flagged as warnings)")
    
    quality_score = 100 * (1 - (gaps + duplicates + invalid) / max(total_rows, 1))
    print(f"[{df_combined['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}] DATA QUALITY: {quality_score:.2f}% [OK]")
    
    # 5. Retornar limpio (only required columns)
    df_clean = df_combined[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    processed_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return df_clean

if __name__ == "__main__":
    validate_data()
