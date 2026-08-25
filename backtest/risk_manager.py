def calculate_position_size(capital_available, risk_per_trade_pct, stop_loss_distance):
    """Calcula el tamaño de la posición basándose en el riesgo."""
    risk_amount = capital_available * risk_per_trade_pct
    if stop_loss_distance <= 0:
        return 0.0
    # The size in base asset (e.g. BTC)
    size = risk_amount / stop_loss_distance
    # Ensure we don't buy more than we can afford (leverage considerations aside)
    # If leverage is used, max_size = (capital * leverage) / price, but we assume risk_amount is the limiting factor.
    return size

def calculate_stop_loss(entry_price, atr, atr_mult, action):
    if action == "BUY":
        return entry_price - (atr * atr_mult)
    elif action == "SELL":
        return entry_price + (atr * atr_mult)
    return entry_price

def calculate_take_profit(entry_price, atr, atr_mult, action):
    if action == "BUY":
        return entry_price + (atr * atr_mult)
    elif action == "SELL":
        return entry_price - (atr * atr_mult)
    return entry_price
