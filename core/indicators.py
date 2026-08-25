
import numpy as np
import pandas as pd


# ---------- 2. TENDENCIA ----------

def sma(df: pd.DataFrame, period: int = 50, col: str = "close") -> pd.Series:
    return df[col].rolling(period).mean()


def ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0

    tr = _true_range(df)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ---------- 3. MOMENTUM ----------

def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"%K": k, "%D": d})


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev)


# ---------- 4. VOLUMEN ----------

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def volume_spike(df: pd.DataFrame, period: int = 20, mult: float = 1.8) -> pd.Series:
    avg_vol = df["volume"].rolling(period).mean()
    return df["volume"] > (avg_vol * mult)


# ---------- 5. ESTRUCTURA DE PRECIO ----------

def support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Máximos y mínimos locales como proxies de resistencia/soporte."""
    resistance = df["high"].rolling(window, center=True).max()
    support = df["low"].rolling(window, center=True).min()
    return pd.DataFrame({"resistance": resistance, "support": support})


def fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> dict:
    recent = df.tail(lookback)
    high, low = recent["high"].max(), recent["low"].min()
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "1.0": low,
    }


def trendline_slope(df: pd.DataFrame, period: int = 20, col: str = "close") -> float:
    """Pendiente de la regresión lineal reciente: positiva = tendencia alcista."""
    y = df[col].tail(period).values
    x = np.arange(len(y))
    if len(y) < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


# ---------- 6. VOLATILIDAD ----------

def bollinger_bands(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = sma(df, period)
    std = df["close"].rolling(period).std()
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": mid + std_mult * std,
        "bb_lower": mid - std_mult * std,
    })


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Añade todas las columnas de indicadores al DataFrame de velas."""
    out = df.copy()
    out["sma_50"] = sma(df, 50)
    out["sma_200"] = sma(df, 200)
    out["ema_20"] = ema(df, 20)
    macd_df = macd(df)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd_df["macd"], macd_df["signal"], macd_df["hist"]
    out["adx"] = adx(df)
    out["rsi"] = rsi(df)
    stoch_df = stochastic(df)
    out["stoch_k"], out["stoch_d"] = stoch_df["%K"], stoch_df["%D"]
    out["cci"] = cci(df)
    out["obv"] = obv(df)
    out["vwap"] = vwap(df)
    out["volume_spike"] = volume_spike(df)
    sr = support_resistance(df)
    out["resistance"], out["support"] = sr["resistance"], sr["support"]
    bb = bollinger_bands(df)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bb["bb_mid"], bb["bb_upper"], bb["bb_lower"]
    out["atr"] = atr(df)
    return out