"""
Genera un live_state.json con datos ficticios para probar el dashboard
sin necesidad de tener el bot corriendo ni conexión a Binance.

Uso:
    python generate_test_state.py
"""
import json
import os
import random
from datetime import datetime, timedelta

from core.live_state import write_state

# Generar ~150 velas ficticias de 15m
base_time = datetime.utcnow() - timedelta(minutes=150 * 15)
price = 60000.0
klines = []
for i in range(150):
    t = base_time + timedelta(minutes=i * 15)
    change = random.uniform(-80, 80)
    o = price
    c = price + change
    h = max(o, c) + random.uniform(10, 60)
    l = min(o, c) - random.uniform(10, 60)
    vol = random.uniform(100, 2000)

    # Indicadores simulados (solo los que usa el dashboard)
    sma_50 = price + random.uniform(-200, 200) if i > 50 else 0
    ema_20 = price + random.uniform(-100, 100) if i > 20 else 0
    bb_upper = price + 500 + random.uniform(-50, 50) if i > 20 else 0
    bb_lower = price - 500 + random.uniform(-50, 50) if i > 20 else 0

    klines.append({
        "open_time": str(t),
        "open": round(o, 2), "high": round(h, 2),
        "low": round(l, 2), "close": round(c, 2),
        "volume": round(vol, 2),
        "sma_50": round(sma_50, 2),
        "ema_20": round(ema_20, 2),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
    })
    price = c

# Trades de ejemplo
trade_history = [
    {"action": "BUY", "price": 59800, "time": str(base_time + timedelta(minutes=30*15)), "pnl": None, "reason": "SIGNAL"},
    {"action": "SELL", "price": 60200, "time": str(base_time + timedelta(minutes=50*15)), "pnl": 1.23, "reason": "TAKE_PROFIT"},
    {"action": "BUY", "price": 60100, "time": str(base_time + timedelta(minutes=80*15)), "pnl": None, "reason": "SIGNAL"},
    {"action": "SELL", "price": 59900, "time": str(base_time + timedelta(minutes=100*15)), "pnl": -0.45, "reason": "STOP_LOSS"},
    {"action": "BUY", "price": 60050, "time": str(base_time + timedelta(minutes=120*15)), "pnl": None, "reason": "SIGNAL"},
]

# Equity curve
equity_curve = []
eq = 15.0
for i in range(100):
    t = base_time + timedelta(minutes=i * 15)
    eq += random.uniform(-0.1, 0.15)
    equity_curve.append({"time": str(t), "equity": round(eq, 4)})

# Posición abierta actual
open_position = {
    "type": "LONG",
    "entry_price": 60050.00,
    "stop_loss": 59700.00,
    "take_profit": 60800.00,
    "liquidation_price": 40033.33,
    "margin_used": 4.80,
}

state = {
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "klines": klines,
    "open_position": open_position,
    "last_signal": {
        "action": "BUY",
        "confirmations": 7,
        "reasons": [
            "sma_cross: +1", "ema_trend: +1", "macd: +1",
            "rsi: +1", "obv: +1", "vwap: +1", "trendline: +1",
        ],
    },
    "trade_history": trade_history,
    "equity_curve": equity_curve,
    "session_info": {
        "start_time": str(base_time),
        "time_remaining": "02:15:30",
        "mode": "ALERTA",
        "capital": round(eq, 2),
    },
}

write_state(state)
print(f"[OK] Estado de prueba escrito en data/live_state.json")
print(f"   Velas: {len(klines)} | Trades: {len(trade_history)} | Equity points: {len(equity_curve)}")
print(f"\n   Ahora corre: python run_dash.py")
print(f"   y abre: http://localhost:8050")
