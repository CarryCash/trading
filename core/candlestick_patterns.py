
import pandas as pd


def _body(row) -> float:
    return abs(row["close"] - row["open"])


def _range(row) -> float:
    return row["high"] - row["low"] if row["high"] != row["low"] else 1e-9


def is_doji(df: pd.DataFrame, threshold: float = 0.1) -> bool:
    row = df.iloc[-1]
    return _body(row) <= threshold * _range(row)


def is_hammer(df: pd.DataFrame) -> bool:
    """Martillo: cuerpo pequeño arriba, mecha inferior larga -> posible reversión alcista."""
    row = df.iloc[-1]
    body = _body(row)
    lower_wick = min(row["open"], row["close"]) - row["low"]
    upper_wick = row["high"] - max(row["open"], row["close"])
    return lower_wick > 2 * body and upper_wick < body


def is_shooting_star(df: pd.DataFrame) -> bool:
    """Estrella fugaz: cuerpo pequeño abajo, mecha superior larga -> posible reversión bajista."""
    row = df.iloc[-1]
    body = _body(row)
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    return upper_wick > 2 * body and lower_wick < body


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] < prev["open"]
        and curr["close"] > curr["open"]
        and curr["close"] >= prev["open"]
        and curr["open"] <= prev["close"]
    )


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] > prev["open"]
        and curr["close"] < curr["open"]
        and curr["open"] >= prev["close"]
        and curr["close"] <= prev["open"]
    )


def is_three_white_soldiers(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    last3 = df.iloc[-3:]
    return bool(
        (last3["close"] > last3["open"]).all()
        and last3["close"].is_monotonic_increasing
    )


def is_three_black_crows(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    last3 = df.iloc[-3:]
    return bool(
        (last3["close"] < last3["open"]).all()
        and last3["close"].is_monotonic_decreasing
    )


def detect_all(df: pd.DataFrame) -> dict:
    """Devuelve un dict {patrón: bool} evaluado sobre la última vela."""
    return {
        "doji": is_doji(df),
        "hammer": is_hammer(df),
        "shooting_star": is_shooting_star(df),
        "bullish_engulfing": is_bullish_engulfing(df),
        "bearish_engulfing": is_bearish_engulfing(df),
        "three_white_soldiers": is_three_white_soldiers(df),
        "three_black_crows": is_three_black_crows(df),
    }