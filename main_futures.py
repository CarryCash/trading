"""
Punto de entrada del bot de FUTURES con validación MTF, alertas (Windows Toasts) 
y exportación de estado para un dashboard local.
"""
import asyncio
import sys
from datetime import datetime, timedelta

import pandas as pd
# pyrefly: ignore [missing-import]
from win10toast import ToastNotifier

from config import Config
from core.binance_future import BinanceFuturesMarketClient, BinanceFuturesStreamClient
from core.indicators import compute_all
from core.future_risk import FuturesRiskManager
from core.live_state import write_state
from strategy.signal_engine import evaluate_mtf

SESSION_DURATION_MINUTES = 240  # 4 horas

# --- Límites para evitar crecimiento indefinido en sesiones 24/7 ---
MAX_TRADE_HISTORY = 200    # conservar últimos N trades
MAX_EQUITY_POINTS = 2000   # conservar últimos N puntos de equity

class FuturesTradingSession:
    def __init__(self):
        self.market = BinanceFuturesMarketClient()
        self.risk = FuturesRiskManager()
        self.session_start = datetime.utcnow()
        self.session_end = self.session_start + timedelta(minutes=SESSION_DURATION_MINUTES)
        self.open_position = None
        self.histories = {}
        self.toaster = ToastNotifier()
        self.trade_history: list = []
        self.equity_curve: list = []
        self.last_signal: dict = {"action": "HOLD", "confirmations": 0, "reasons": []}

    def _confirm_live_trading(self):
        if Config.ALERT_MODE:
            print("=" * 60)
            print("MODO ALERTA ACTIVADO: El bot NO ejecutará trades reales, solo notificará.")
            print("=" * 60)
            return

        if Config.TESTNET or not Config.REQUIRE_LIVE_CONFIRMATION:
            return
        print("=" * 60)
        print("ATENCIÓN: FUTURES con DINERO REAL y apalancamiento.")
        print(f"Símbolo: {Config.SYMBOL} | Leverage: {Config.LEVERAGE}x")
        print("=" * 60)
        answer = input("Escribe exactamente CONFIRMO para continuar: ").strip()
        if answer != "CONFIRMO":
            print("No confirmado. Cerrando sin operar.")
            sys.exit(0)

    def _load_initial_history(self):
        print("Cargando historial MTF...")
        timeframes_to_load = set(Config.TIMEFRAMES)
        timeframes_to_load.add(Config.PRIMARY_TIMEFRAME)
        for tf in timeframes_to_load:
            df = self.market.get_klines_df(Config.SYMBOL, tf, limit=300)
            self.histories[tf] = compute_all(df)
        print("Historial cargado.")

    def _session_time_left(self) -> timedelta:
        return self.session_end - datetime.utcnow()
    
    def _export_dashboard_state(self):
        try:
            primary_df = self.histories.get(Config.PRIMARY_TIMEFRAME)
            if primary_df is None or primary_df.empty:
                return
                
            # Tomamos las últimas 150 velas para el gráfico
            recent = primary_df.tail(150).copy()
            recent["open_time"] = recent["open_time"].astype(str)
            recent = recent.fillna(0)

            remaining = self._session_time_left()
            
            data = {
                "symbol": Config.SYMBOL,
                "timeframe": Config.PRIMARY_TIMEFRAME,
                "klines": recent.to_dict(orient="records"),
                "open_position": self.open_position,
                "last_signal": self.last_signal,
                "trade_history": self.trade_history[-MAX_TRADE_HISTORY:],
                "equity_curve": self.equity_curve[-MAX_EQUITY_POINTS:],
                "session_info": {
                    "start_time": str(self.session_start),
                    "time_remaining": str(remaining).split(".")[0] if remaining.total_seconds() > 0 else "00:00:00",
                    "mode": "ALERTA" if Config.ALERT_MODE else "REAL",
                    "capital": self.risk.capital,
                }
            }
            write_state(data)
        except Exception as e:
            print(f"[Error Exportando Dashboard] {e}")

    def _trigger_alert(self, action: str, price: float, result):
        title = f"SEÑAL {action} en {Config.SYMBOL}"
        msg = f"Precio: {price:.2f}\nConfirmaciones: {result.confirmations}\nValidación MTF Pasada"
        print(f"\n>>> ALERTA: {title} - {msg} <<<")
        
        try:
            self.toaster.show_toast(title, msg, duration=10, threaded=True)
        except Exception as e:
            print(f"[Error Toast] {e}")

    async def _on_kline_close(self, kline: dict):
        if datetime.utcnow() >= self.session_end:
            print("\n[SESIÓN] Tiempo agotado. Cerrando bot.")
            raise SystemExit(0)

        # Actualizamos la vela actual (PRIMARY)
        primary = Config.PRIMARY_TIMEFRAME
        new_row = pd.DataFrame([{
            "open_time": pd.to_datetime(kline["t"], unit="ms"),
            "open": float(kline["o"]), "high": float(kline["h"]),
            "low": float(kline["l"]), "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }])
        self.histories[primary] = pd.concat([self.histories[primary], new_row], ignore_index=True).tail(500)
        self.histories[primary] = compute_all(self.histories[primary])

        # Actualizamos otros timeframes (REST request rapido en el cierre de vela)
        for tf in Config.TIMEFRAMES:
            if tf != primary:
                df = self.market.get_klines_df(Config.SYMBOL, tf, limit=300)
                self.histories[tf] = compute_all(df)

        self._export_dashboard_state()

        result = evaluate_mtf(self.histories, primary)
        price = self.histories[primary].iloc[-1]["close"]
        remaining = self._session_time_left()

        # Guardar última señal y punto de equity
        self.last_signal = {
            "action": result.action,
            "confirmations": result.confirmations,
            "reasons": result.reasons,
        }
        self.equity_curve.append({
            "time": str(datetime.utcnow()),
            "equity": self.risk.capital,
        })
        
        print(f"[{datetime.utcnow():%H:%M:%S}] precio={price:.2f} señal={result.action} "
              f"confirmaciones={result.confirmations} mtfs={'OK' if result.action != 'HOLD' else '-'} time_left={remaining}")

        if Config.ALERT_MODE:
            if result.action in ["BUY", "SELL"]:
                self._trigger_alert(result.action, price, result)
        else:
            if self.risk.daily_loss_limit_hit():
                print("[RIESGO] Límite de pérdida diaria alcanzado. No se abren nuevas posiciones.")
                return

            if result.action == "BUY" and self.open_position is None:
                self._execute_long(price)
            elif result.action == "SELL" and self.open_position is not None:
                self._execute_close(price)

    def _execute_long(self, price: float):
        atr_value = self.histories[Config.PRIMARY_TIMEFRAME].iloc[-1]["atr"]
        plan = self.risk.plan_long_position(price, atr_value)
        if plan.quantity <= 0:
            return
        if not self.risk.stop_is_safe_vs_liquidation(plan, is_long=True):
            print("[BLOQUEADO] El stop-loss quedaría más allá del precio de liquidación.")
            return

        print(f"[ORDEN] LONG {plan.quantity:.5f} {Config.SYMBOL} a ~{price:.2f} "
              f"| margen={plan.margin_used:.2f} USDT | stop={plan.stop_loss:.2f} ")
        try:
            self.market.ensure_leverage_and_margin(Config.SYMBOL)
            self.market.place_market_order(Config.SYMBOL, "BUY", plan.quantity)
            self.market.place_stop_loss(Config.SYMBOL, "SELL", plan.quantity, plan.stop_loss)
            self.market.place_take_profit(Config.SYMBOL, "SELL", plan.quantity, plan.take_profit)
            self.open_position = {
                "type": "LONG",
                "entry_price": plan.entry_price,
                "stop_loss": plan.stop_loss,
                "take_profit": plan.take_profit,
                "liquidation_price": plan.liquidation_price,
                "margin_used": plan.margin_used,
            }
            self.trade_history.append({
                "action": "BUY", "price": price,
                "time": str(datetime.utcnow()), "pnl": None, "reason": "SIGNAL",
            })
            self._export_dashboard_state()
        except Exception as e:
            print(f"[ERROR AL EJECUTAR ORDEN]: {e}")

    def _execute_close(self, price: float, reason: str = "SEÑAL_CONTRARIA"):
        entry_price = self.open_position["entry_price"]
        # Calcular PnL aproximado usando el margen y el movimiento del precio
        margin_used = self.open_position["margin_used"]
        pnl = (price - entry_price) / entry_price * margin_used * Config.LEVERAGE
        try:
            self.market.cancel_all_open_orders(Config.SYMBOL)
            # Cantidad se calcula del margen original
            qty = margin_used * Config.LEVERAGE / entry_price
            self.market.place_market_order(Config.SYMBOL, "SELL", qty)
            self.risk.register_trade_result(pnl)
            self.risk.capital += pnl
            print(f"[ORDEN] Cierre: {reason}. PnL aprox: {pnl:.4f} USDT")
        except Exception as e:
            print(f"[ERROR AL CERRAR POSICIÓN]: {e}")
        self.trade_history.append({
            "action": "SELL", "price": price,
            "time": str(datetime.utcnow()), "pnl": pnl, "reason": reason,
        })
        self.open_position = None
        self._export_dashboard_state()

    async def start(self):
        self._confirm_live_trading()
        if not Config.ALERT_MODE:
            self.market.ensure_leverage_and_margin(Config.SYMBOL)
        self._load_initial_history()
        self._export_dashboard_state()
        print(f"[SESIÓN INICIADA] {Config.SYMBOL} | modo={'ALERTA' if Config.ALERT_MODE else 'REAL'} "
              f"| tf={Config.PRIMARY_TIMEFRAME}")
        stream = BinanceFuturesStreamClient(Config.SYMBOL, Config.PRIMARY_TIMEFRAME, self._on_kline_close)
        try:
            await stream.run()
        except SystemExit:
            pass
        print("[FIN DE SESIÓN]")

if __name__ == "__main__":
    asyncio.run(FuturesTradingSession().start())