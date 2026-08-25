"""
Gestión de riesgo para FUTURES. La diferencia clave frente a spot: el tamaño de la
posición ahora se calcula sobre MARGEN (lo que realmente arriesgas), no sobre el
valor completo de la posición — y siempre calculamos el precio de liquidación
aproximado para saber cuánto margen de seguridad hay respecto al stop-loss.
"""
from dataclasses import dataclass

from config import Config


@dataclass
class FuturesPositionPlan:
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float          # tamaño de la posición en el activo (ej. BTC)
    margin_used: float       # USDT realmente bloqueados como margen
    notional: float          # valor total de la posición (margin_used * leverage)
    liquidation_price: float # precio aproximado de liquidación


class FuturesRiskManager:
    def __init__(self, capital_usdt: float = None):
        self.capital = capital_usdt or Config.STARTING_CAPITAL_USDT
        self.daily_pnl = 0.0
        self.trades_today = 0

    def daily_loss_limit_hit(self) -> bool:
        return self.daily_pnl <= -(self.capital * Config.MAX_DAILY_LOSS_PCT)

    def _estimate_liquidation_price(self, entry_price: float, is_long: bool) -> float:
        """
        Aproximación simplificada (margen aislado, sin comisiones):
        precio de liquidación ≈ entrada * (1 -+ 1/leverage)
        El valor real de Binance varía un poco por el maintenance margin rate,
        pero esto da una referencia conservadora.
        """
        leverage = Config.LEVERAGE
        if is_long:
            return entry_price * (1 - 1 / leverage)
        return entry_price * (1 + 1 / leverage)

    def plan_long_position(self, entry_price: float, atr_value: float) -> FuturesPositionPlan:
        risk_usdt = self.capital * Config.RISK_PER_TRADE_PCT
        stop_distance = atr_value * Config.STOP_LOSS_ATR_MULT
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (atr_value * Config.TAKE_PROFIT_ATR_MULT)

        # Cantidad para que, si toca el stop, la pérdida sea exactamente risk_usdt
        quantity = risk_usdt / stop_distance if stop_distance > 0 else 0

        notional = quantity * entry_price
        margin_used = notional / Config.LEVERAGE
        # Nunca usar más margen del capital disponible
        if margin_used > self.capital:
            scale = self.capital / margin_used
            quantity *= scale
            notional = quantity * entry_price
            margin_used = notional / Config.LEVERAGE

        liq_price = self._estimate_liquidation_price(entry_price, is_long=True)

        return FuturesPositionPlan(entry_price, stop_loss, take_profit, quantity,
                                    margin_used, notional, liq_price)

    def plan_short_position(self, entry_price: float, atr_value: float) -> FuturesPositionPlan:
        risk_usdt = self.capital * Config.RISK_PER_TRADE_PCT
        stop_distance = atr_value * Config.STOP_LOSS_ATR_MULT
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - (atr_value * Config.TAKE_PROFIT_ATR_MULT)

        quantity = risk_usdt / stop_distance if stop_distance > 0 else 0
        notional = quantity * entry_price
        margin_used = notional / Config.LEVERAGE
        if margin_used > self.capital:
            scale = self.capital / margin_used
            quantity *= scale
            notional = quantity * entry_price
            margin_used = notional / Config.LEVERAGE

        liq_price = self._estimate_liquidation_price(entry_price, is_long=False)

        return FuturesPositionPlan(entry_price, stop_loss, take_profit, quantity,
                                    margin_used, notional, liq_price)

    def stop_is_safe_vs_liquidation(self, plan: FuturesPositionPlan, is_long: bool) -> bool:
        """Verifica que el stop-loss se ejecute ANTES de llegar al precio de liquidación."""
        if is_long:
            return plan.stop_loss > plan.liquidation_price
        return plan.stop_loss < plan.liquidation_price

    def register_trade_result(self, pnl_usdt: float):
        self.daily_pnl += pnl_usdt
        self.trades_today += 1

    def can_open_new_position(self, open_positions: int) -> bool:
        if self.daily_loss_limit_hit():
            return False
        if open_positions >= Config.MAX_OPEN_POSITIONS:
            return False
        return True