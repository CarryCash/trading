"""
Configuración central del bot.
NUNCA pongas tus API keys directamente en este archivo.
Usa un archivo .env (ver .env.example) que NO subas a ningún repositorio público.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Modo de operación ---
    # TESTNET = True  -> usa la red de pruebas de Binance (dinero ficticio). SIEMPRE empezar aquí.
    # TESTNET = False -> usa Binance real, dinero real. Requiere confirmación explícita en runtime.
    TESTNET: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    ALERT_MODE: bool = True  # If True, bot only sends alerts and does not execute trades


    # --- Credenciales (se leen del entorno, nunca hardcodeadas) ---
    API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY" if TESTNET else "BINANCE_API_KEY", "")
    API_SECRET: str = os.getenv("BINANCE_TESTNET_API_SECRET" if TESTNET else "BINANCE_API_SECRET", "")

    # --- Mercado ---
    MARKET_TYPE: str = os.getenv("MARKET_TYPE", "futures")  # "spot" o "futures"
    SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT")

    # --- Futures: leverage y margen ---
    # LEVERAGE bajo (2x-3x) = un movimiento en contra necesita ser grande para liquidarte.
    # LEVERAGE alto (10x+) = pequeños movimientos en contra pueden liquidarte por completo.
    LEVERAGE: int = int(os.getenv("LEVERAGE", "3"))
    MARGIN_TYPE: str = os.getenv("MARGIN_TYPE", "ISOLATED")  # ISOLATED: pérdida máxima = margen de esa posición
    TIMEFRAMES: list = ["1M", "1w", "1d", "1h", "15m", "5m", "1m"]  # multi-timeframe, mes -> 1 min
    PRIMARY_TIMEFRAME: str = "15m"  # timeframe donde se generan las señales de entrada
    TREND_TIMEFRAME: str = "4h"      # timeframe para confirmar tendencia general

    # --- Gestión de riesgo (CRÍTICO con capital pequeño) ---
    STARTING_CAPITAL_USDT: float = float(os.getenv("STARTING_CAPITAL_USDT", "15"))
    RISK_PER_TRADE_PCT: float = 0.015       # con leverage, arriesgar menos % por trade (el leverage ya amplifica)
    MAX_DAILY_LOSS_PCT: float = 0.10        # el bot se detiene solo si pierde 10% del capital en el día
    
    # El ratio de Riesgo/Beneficio está dictado por los multiplicadores de ATR.
    # SL 0.8 × TP 3.0 = ratio 1:3.75. Con 56% WR:
    #   EV = (0.56 × 3.0) - (0.44 × 0.8) = 1.68 - 0.352 = +1.328 (muy viable después de comisiones)
    STOP_LOSS_ATR_MULT: float = 0.8         # stop loss = 0.8x el ATR (reducido para mejorar EV)
    TAKE_PROFIT_ATR_MULT: float = 3.0       # take profit = 3.0x el ATR (ratio riesgo/beneficio 1:3.75)
    MAX_OPEN_POSITIONS: int = 1             # con $15, una sola posición a la vez

    # --- Umbral de decisión ---
    # Cuántas de las señales (de las ~18 herramientas) deben coincidir para actuar.
    # NOTA: 5 confirmaciones es un umbral permisivo (~27% de las señales). 
    # Para mayor rigurosidad se recomienda 7 o 9, pero se mantiene en 5 a petición 
    # inicial para probar el flujo de señales sin asfixiar la estrategia.
    MIN_CONFIRMATIONS_TO_TRADE: int = 5

    # --- Seguridad ---
    REQUIRE_LIVE_CONFIRMATION: bool = True  # pide "SI CONFIRMO" en consola antes de operar con dinero real