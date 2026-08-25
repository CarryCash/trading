import os
import sys
import sqlite3
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DB_PATH = os.path.join(os.path.dirname(__file__), 'market_data.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME UNIQUE NOT NULL,
        close REAL,
        sma_50 REAL,
        sma_200 REAL,
        ema_20 REAL,
        rsi REAL,
        stoch_k REAL,
        cci REAL,
        macd REAL,
        macd_hist REAL,
        obv REAL,
        vwap REAL,
        bb_upper REAL,
        bb_lower REAL,
        atr REAL,
        support REAL,
        resistance REAL,
        trendline_slope REAL,
        fibonacci_618 REAL,
        bollinger INTEGER,
        pattern_hammer INTEGER,
        pattern_engulfing INTEGER,
        pattern_three_soldiers INTEGER,
        LABEL INTEGER
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON features(timestamp)')
    conn.commit()
    conn.close()

def insert_features(df_features):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    # Rename columns to match schema if necessary, or just subset
    cols = [
        'timestamp', 'close', 'sma_50', 'sma_200', 'ema_20', 'rsi', 'stoch_k', 'cci',
        'macd', 'macd_hist', 'obv', 'vwap', 'bb_upper', 'bb_lower', 'atr', 'support',
        'resistance', 'pattern_hammer', 'pattern_engulfing', 'pattern_three_soldiers', 'LABEL'
    ]
    # some columns might not exist precisely as named, e.g. fibonacci_618, bollinger, trendline_slope.
    # Let's add them as dummy or calculate if missing to match schema perfectly
    if 'fibonacci_618' not in df_features.columns:
        df_features['fibonacci_618'] = 0.0
    if 'bollinger' not in df_features.columns:
        # Based on bb_upper and bb_lower
        df_features['bollinger'] = 0
    if 'trendline_slope' not in df_features.columns:
        df_features['trendline_slope'] = 0.0
        
    subset = df_features[[c for c in cols if c in df_features.columns] + ['fibonacci_618', 'bollinger', 'trendline_slope']]
    subset = subset.loc[:, ~subset.columns.duplicated()]
    
    try:
        subset.to_sql('features', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON features(timestamp)')
        print(f"[OK] {len(subset)} filas insertadas en market_data.db")
    except Exception as e:
        print(f"Error inserting: {e}")
    finally:
        conn.commit()
        conn.close()

def get_range(start, end):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM features WHERE timestamp >= ? AND timestamp <= ?"
    df = pd.read_sql(query, conn, params=(start, end))
    conn.close()
    return df

def get_latest(n):
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM features ORDER BY timestamp DESC LIMIT {n}"
    df = pd.read_sql(query, conn)
    # Reverse to keep chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    conn.close()
    return df

def get_label_distribution():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT LABEL, COUNT(*) as count
    FROM features
    GROUP BY LABEL
    """
    try:
        result = pd.read_sql(query, conn)
    except:
        result = pd.DataFrame()
    conn.close()
    return result

if __name__ == "__main__":
    processed_dir = os.path.join(os.path.dirname(__file__), "processed")
    features_csv = os.path.join(processed_dir, 'features_balanced.csv')
    if os.path.exists(features_csv):
        df = pd.read_csv(features_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        insert_features(df)
        dist = get_label_distribution()
        print("\nDistribución de Labels:")
        print(dist.to_string(index=False))
    else:
        print(f"Features file not found at {features_csv}")
