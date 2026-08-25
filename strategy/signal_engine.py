import logging
from dataclasses import dataclass, field
import pandas as pd

from core import indicators as ind
from core import candlestick_patterns as cdl
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    action: str                # "BUY" | "SELL" | "HOLD"
    score: int                 # suma neta de votos
    confirmations: int         # cuántas señales coincidieron con la dirección elegida
    votes: dict = field(default_factory=dict)   # detalle de cada herramienta -> voto
    reasons: list = field(default_factory=list) # explicación legible


def _vote_trend(row: pd.Series) -> dict:
    """Evalúa indicadores de tendencia."""
    votes = {}
    # 1. SMA50 vs SMA200 (cruce dorado/muerte simplificado)
    votes["sma_cross"] = 1 if row["sma_50"] > row["sma_200"] else -1
    # 2. EMA20 vs precio
    votes["ema_trend"] = 1 if row["close"] > row["ema_20"] else -1
    # 3. MACD histograma
    votes["macd"] = 1 if row["macd_hist"] > 0 else -1
    # ADX solo confirma fuerza, no dirección: se usa como filtro, no como voto direccional
    return votes


def _vote_momentum(row: pd.Series) -> dict:
    """Evalúa indicadores de momentum."""
    votes = {}
    # 4. RSI
    if row["rsi"] < 30:
        votes["rsi"] = 1
    elif row["rsi"] > 70:
        votes["rsi"] = -1
    else:
        votes["rsi"] = 0
        
    # 5. Estocástico
    if row["stoch_k"] < 20:
        votes["stochastic"] = 1
    elif row["stoch_k"] > 80:
        votes["stochastic"] = -1
    else:
        votes["stochastic"] = 0
        
    # 6. CCI
    if row["cci"] < -100:
        votes["cci"] = 1
    elif row["cci"] > 100:
        votes["cci"] = -1
    else:
        votes["cci"] = 0
    return votes


def _vote_volume(df: pd.DataFrame) -> dict:
    """Evalúa indicadores basados en volumen."""
    votes = {}
    row = df.iloc[-1]
    prev_obv = df["obv"].iloc[-2] if len(df) > 1 else row["obv"]
    # 7. OBV creciente = presión compradora
    votes["obv"] = 1 if row["obv"] > prev_obv else -1
    # 8. Precio vs VWAP
    votes["vwap"] = 1 if row["close"] > row["vwap"] else -1
    # 9. Volume spike (confirma fuerza del movimiento actual, no dirección propia)
    votes["volume_spike"] = 1 if row["volume_spike"] and row["close"] > row["open"] else (
        -1 if row["volume_spike"] and row["close"] < row["open"] else 0
    )
    return votes


def _vote_structure(df: pd.DataFrame) -> dict:
    """Evalúa la estructura del precio (soportes, resistencias, fibonacci)."""
    votes = {}
    row = df.iloc[-1]
    # 10. Cercanía a soporte/resistencia
    if pd.notna(row["support"]) and row["close"] <= row["support"] * 1.005:
        votes["support_resistance"] = 1
    elif pd.notna(row["resistance"]) and row["close"] >= row["resistance"] * 0.995:
        votes["support_resistance"] = -1
    else:
        votes["support_resistance"] = 0
        
    # 11. Pendiente de la línea de tendencia
    slope = ind.trendline_slope(df)
    votes["trendline"] = 1 if slope > 0 else -1
    
    # 12. Fibonacci: cerca del 0.618 (zona de rebote clásica)
    fib = ind.fibonacci_levels(df)
    price = row["close"]
    near_618 = abs(price - fib["0.618"]) / price < 0.005
    votes["fibonacci"] = 1 if near_618 and price > fib["1.0"] else 0
    return votes


def _vote_volatility(row: pd.Series) -> dict:
    """Evalúa indicadores de volatilidad."""
    votes = {}
    # 13. Bandas de Bollinger
    if row["close"] <= row["bb_lower"]:
        votes["bollinger"] = 1
    elif row["close"] >= row["bb_upper"]:
        votes["bollinger"] = -1
    else:
        votes["bollinger"] = 0
    # ATR no vota dirección, se usa para el tamaño de stop/take-profit (risk_manager.py)
    return votes


def _vote_candles(df: pd.DataFrame) -> dict:
    """Evalúa patrones de velas japonesas."""
    patterns = cdl.detect_all(df)
    votes = {}
    # 14-18. Patrones de velas
    votes["hammer"] = 1 if patterns["hammer"] else 0
    votes["shooting_star"] = -1 if patterns["shooting_star"] else 0
    votes["bullish_engulfing"] = 1 if patterns["bullish_engulfing"] else 0
    votes["bearish_engulfing"] = -1 if patterns["bearish_engulfing"] else 0
    votes["three_soldiers_crows"] = (
        1 if patterns["three_white_soldiers"] else (-1 if patterns["three_black_crows"] else 0)
    )
    return votes


def evaluate_mtf(dfs: dict, primary_tf: str) -> SignalResult:
    """
    Recibe un diccionario de DataFrames procesados por indicators.compute_all() para cada timeframe.
    Evalúa las señales en el timeframe principal y filtra según la tendencia en timeframes superiores.
    """
    if primary_tf not in dfs:
        logger.error(f"Falta timeframe principal '{primary_tf}' en los datos.")
        return SignalResult("HOLD", 0, 0, {}, ["Falta timeframe principal"])
    
    df_primary = dfs[primary_tf]
    if len(df_primary) == 0:
        logger.warning(f"El DataFrame del timeframe principal '{primary_tf}' está vacío.")
        return SignalResult("HOLD", 0, 0, {}, ["DataFrame principal vacío"])

    row = df_primary.iloc[-1]

    all_votes = {}
    all_votes.update(_vote_trend(row))
    all_votes.update(_vote_momentum(row))
    all_votes.update(_vote_volume(df_primary))
    all_votes.update(_vote_structure(df_primary))
    all_votes.update(_vote_volatility(row))
    all_votes.update(_vote_candles(df_primary))

    score = sum(all_votes.values())
    bullish_confirmations = sum(1 for v in all_votes.values() if v > 0)
    bearish_confirmations = sum(1 for v in all_votes.values() if v < 0)

    # Validacion Multi-Timeframe (MTF)
    mtf_bullish = True
    mtf_bearish = True
    for tf, df_tf in dfs.items():
        if tf == primary_tf or len(df_tf) == 0:
            continue
        # Un filtro simple: usar la SMA50 del timeframe mayor para confirmar la tendencia general
        row_tf = df_tf.iloc[-1]
        if row_tf["close"] < row_tf["sma_50"]:
            mtf_bullish = False
        if row_tf["close"] > row_tf["sma_50"]:
            mtf_bearish = False

    # Filtro de fuerza de tendencia (ADX > 20 indica tendencia fuerte)
    weak_trend = row["adx"] < 20
    threshold = Config.MIN_CONFIRMATIONS_TO_TRADE + (1 if weak_trend else 0)

    reasons = [f"{k}: {'+1' if v > 0 else ('-1' if v < 0 else '0')}" for k, v in all_votes.items() if v != 0]
    
    # Añadimos a reasons la validación MTF
    if not mtf_bullish and bullish_confirmations >= threshold:
        msg = "Rechazado (MTF): La tendencia en timeframes superiores no es alcista."
        reasons.append(msg)
        logger.info(f"Señal de COMPRA rechazada: {msg}")
    
    if not mtf_bearish and bearish_confirmations >= threshold:
        msg = "Rechazado (MTF): La tendencia en timeframes superiores no es bajista."
        reasons.append(msg)
        logger.info(f"Señal de VENTA rechazada: {msg}")

    if bullish_confirmations >= threshold and bullish_confirmations > bearish_confirmations and mtf_bullish:
        logger.info(f"Señal BUY generada con {bullish_confirmations} confirmaciones y score {score}")
        return SignalResult("BUY", score, bullish_confirmations, all_votes, reasons)
        
    if bearish_confirmations >= threshold and bearish_confirmations > bullish_confirmations and mtf_bearish:
        logger.info(f"Señal SELL generada con {bearish_confirmations} confirmaciones y score {score}")
        return SignalResult("SELL", score, bearish_confirmations, all_votes, reasons)
        
    logger.debug(f"Señal HOLD generada (Bullish: {bullish_confirmations}, Bearish: {bearish_confirmations}, Thresh: {threshold})")
    return SignalResult("HOLD", score, max(bullish_confirmations, bearish_confirmations), all_votes, reasons)