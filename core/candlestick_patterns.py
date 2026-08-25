import pandas as pd


def _body(row: pd.Series) -> float:
    """Devuelve el tamaño del cuerpo de la vela."""
    return abs(row["close"] - row["open"])


def _range(row: pd.Series) -> float:
    """Devuelve el tamaño total de la vela (mechas incluidas). Evita división por cero."""
    return row["high"] - row["low"] if row["high"] != row["low"] else 1e-9


def is_doji(df: pd.DataFrame, threshold: float = 0.1) -> bool:
    """
    Doji: Apertura y cierre casi idénticos. Indica indecisión.
    :param threshold: Tolerancia del cuerpo respecto al tamaño total de la vela.
    """
    row = df.iloc[-1]
    return _body(row) <= threshold * _range(row)


def is_hammer(df: pd.DataFrame) -> bool:
    """
    Martillo (Hammer): Cuerpo pequeño arriba, mecha inferior larga.
    Indica posible reversión alcista al encontrar soporte.
    """
    row = df.iloc[-1]
    body = _body(row)
    lower_wick = min(row["open"], row["close"]) - row["low"]
    upper_wick = row["high"] - max(row["open"], row["close"])
    return lower_wick > 2 * body and upper_wick < body


def is_shooting_star(df: pd.DataFrame) -> bool:
    """
    Estrella fugaz (Shooting Star): Cuerpo pequeño abajo, mecha superior larga.
    Indica posible reversión bajista al encontrar resistencia.
    """
    row = df.iloc[-1]
    body = _body(row)
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    return upper_wick > 2 * body and lower_wick < body


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """
    Envolvente Alcista: La vela verde actual envuelve completamente a la vela roja anterior.
    """
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
    """
    Envolvente Bajista: La vela roja actual envuelve completamente a la vela verde anterior.
    """
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
    """
    Tres soldados blancos: Tres velas verdes consecutivas, cada una con cierre mayor a la anterior.
    Fuerte señal alcista.
    """
    if len(df) < 3:
        return False
    last3 = df.iloc[-3:]
    return bool(
        (last3["close"] > last3["open"]).all()
        and last3["close"].is_monotonic_increasing
    )


def is_three_black_crows(df: pd.DataFrame) -> bool:
    """
    Tres cuervos negros: Tres velas rojas consecutivas, cada una con cierre menor a la anterior.
    Fuerte señal bajista.
    """
    if len(df) < 3:
        return False
    last3 = df.iloc[-3:]
    return bool(
        (last3["close"] < last3["open"]).all()
        and last3["close"].is_monotonic_decreasing
    )


def detect_all(df: pd.DataFrame) -> dict:
    """Devuelve un dict {patrón: bool} evaluado sobre la última vela del DataFrame."""
    return {
        "doji": is_doji(df),
        "hammer": is_hammer(df),
        "shooting_star": is_shooting_star(df),
        "bullish_engulfing": is_bullish_engulfing(df),
        "bearish_engulfing": is_bearish_engulfing(df),
        "three_white_soldiers": is_three_white_soldiers(df),
        "three_black_crows": is_three_black_crows(df),
    }