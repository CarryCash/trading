"""
Backtesting para FUTURES: misma lógica de señales que spot, pero el PnL se calcula
sobre posiciones apalancadas y se verifica que ninguna operación histórica habría
tocado el precio de liquidación antes que el stop-loss.

Uso:
    python backtest_futures.py --limit 2000
"""
import argparse

import pandas as pd

from config import Config
from core.binance_future import BinanceFuturesMarketClient
from core.indicators import compute_all
from core.future_risk import FuturesRiskManager
from strategy.signal_engine import evaluate


def run_futures_backtest(symbol: str, interval: str, limit: int, starting_capital: float):
    client = BinanceFuturesMarketClient()
    raw = client.get_klines_df(symbol, interval, limit=limit)

    risk = FuturesRiskManager(starting_capital)
    equity = starting_capital
    equity_curve = []
    open_position = None
    trades = []
    liquidation_near_misses = 0

    min_window = 210
    for i in range(min_window, len(raw)):
        window = raw.iloc[: i + 1]
        df = compute_all(window)
        row = df.iloc[-1]
        price = row["close"]

        result = evaluate(df)

        if open_position is None and result.action == "BUY" and not risk.daily_loss_limit_hit():
            plan = risk.plan_long_position(price, row["atr"])
            if plan.quantity > 0 and risk.stop_is_safe_vs_liquidation(plan, is_long=True):
                open_position = plan
                trades.append({"type": "BUY", "time": row["open_time"], "price": price,
                                "margin": plan.margin_used, "liq_price": plan.liquidation_price})

        elif open_position is not None:
            hit_liq = price <= open_position.liquidation_price
            hit_stop = price <= open_position.stop_loss
            hit_tp = price >= open_position.take_profit
            signal_exit = result.action == "SELL"

            if hit_liq or hit_stop or hit_tp or signal_exit:
                exit_price = open_position.liquidation_price if hit_liq else price
                pnl = (exit_price - open_position.entry_price) * open_position.quantity
                if hit_liq:
                    pnl = -open_position.margin_used  # liquidación = pierdes el margen completo
                    liquidation_near_misses += 1
                equity += pnl
                risk.register_trade_result(pnl)
                reason = "LIQUIDACION" if hit_liq else ("STOP_LOSS" if hit_stop else
                          ("TAKE_PROFIT" if hit_tp else "SEÑAL_CONTRARIA"))
                trades.append({"type": "SELL", "time": row["open_time"], "price": exit_price,
                                "pnl": pnl, "reason": reason})
                open_position = None

        equity_curve.append({"time": row["open_time"], "equity": equity})

    _print_report(symbol, interval, starting_capital, equity, trades, liquidation_near_misses)
    return pd.DataFrame(equity_curve), pd.DataFrame(trades)


def _print_report(symbol, interval, starting_capital, final_equity, trades, liquidations):
    sells = [t for t in trades if t["type"] == "SELL"]
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]
    total_pnl = final_equity - starting_capital
    win_rate = (len(wins) / len(sells) * 100) if sells else 0

    print("=" * 60)
    print(f"BACKTEST FUTURES: {symbol} | timeframe={interval} | leverage={Config.LEVERAGE}x")
    print("=" * 60)
    print(f"Capital inicial:      {starting_capital:.2f} USDT")
    print(f"Capital final:        {final_equity:.2f} USDT")
    print(f"PnL total:            {total_pnl:+.2f} USDT ({total_pnl/starting_capital*100:+.1f}%)")
    print(f"Operaciones cerradas: {len(sells)}")
    print(f"Ganadas / Perdidas:   {len(wins)} / {len(losses)}")
    print(f"Win rate:             {win_rate:.1f}%")
    print(f"LIQUIDACIONES:        {liquidations}  {'⚠️ revisa tu leverage/stop-loss' if liquidations else ''}")
    print("=" * 60)
    print("NOTA: resultados pasados no garantizan resultados futuros, y esto usa velas")
    print("de CIERRE — en vivo el precio puede tocar liquidación INTRAVELA sin que el")
    print("backtest lo capture exactamente. Trata el número de liquidaciones como piso, no techo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=Config.SYMBOL)
    parser.add_argument("--interval", default=Config.PRIMARY_TIMEFRAME)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--capital", type=float, default=Config.STARTING_CAPITAL_USDT)
    args = parser.parse_args()

    run_futures_backtest(args.symbol, args.interval, args.limit, args.capital)