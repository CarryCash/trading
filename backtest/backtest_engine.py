import numpy as np
import pandas as pd 

class BacktestEngine:
    def __init__(self, initial_capital=15.0, fee_pct=0.001):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct

    def execute_oracle(self, df):
        """
        Backtest basado en Oracle (Label perfecto).
        LABEL: 1 (BUY), -1 (SELL), 0 (HOLD)
        Entra a market, cierra exactamente en 2 velas.
        """
        capital = self.initial_capital
        trades = []
        peak_capital = capital
        max_drawdown = 0.0
        
        # Iteramos hasta len(df) - 2 porque requerimos cierre a t+2
        for t in range(len(df) - 2):
            row = df.iloc[t]
            label = row['LABEL']
            
            if label == 0:
                continue
                
            entry_price = row['close']
            exit_price = df.iloc[t+2]['close']
            entry_time = row['timestamp']
            exit_time = df.iloc[t+2]['timestamp']
            
            # Simplified sizing: bet 50% of capital with 3x leverage = position size = (capital * 0.5 * 3) / price
            # Or just risk a fixed % of capital. Since Oracle wins almost always, let's just bet 100% * leverage 3.
            leverage = 3
            position_size = (capital * leverage) / entry_price
            
            # Fees
            entry_fee = position_size * entry_price * self.fee_pct
            exit_fee = position_size * exit_price * self.fee_pct
            
            if label == 1:
                pnl = (exit_price - entry_price) * position_size
                action = "BUY"
            elif label == -1:
                pnl = (entry_price - exit_price) * position_size
                action = "SELL"
                
            net_pnl = pnl - (entry_fee + exit_fee)
            capital += net_pnl
            
            if capital > peak_capital:
                peak_capital = capital
            
            dd = (peak_capital - capital) / peak_capital * 100
            if dd > max_drawdown:
                max_drawdown = dd
                
            trades.append({
                'entry_time': str(entry_time),
                'entry_price': float(entry_price),
                'entry_signal': action,
                'position_size': float(position_size),
                'exit_time': str(exit_time),
                'exit_price': float(exit_price),
                'pnl_usd': float(net_pnl),
                'pnl_pct': float((net_pnl / (position_size * entry_price)) * 100),
                'win': bool(net_pnl > 0)
            })
            
        return trades, self.calculate_metrics(trades, capital, max_drawdown, df.iloc[0]['timestamp'], df.iloc[-1]['timestamp'], "Oracle")

    def execute_baseline(self, df):
        """
        Backtest basado en el motor de señales original (Baseline).
        Nota: Ya que no tenemos el objeto `signal_engine` corriendo en vivo con DFS mutables,
        vamos a emular las condiciones usando las columnas ya calculadas.
        """
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from strategy.signal_engine import _vote_trend, _vote_momentum, _vote_volatility
        from strategy.signal_engine import _vote_volume, _vote_structure, _vote_candles
        from backtest.risk_manager import calculate_position_size, calculate_stop_loss, calculate_take_profit
        
        capital = self.initial_capital
        trades = []
        
        active_trade = None
        peak_capital = capital
        max_drawdown = 0.0
        
        for t in range(len(df)):
            row = df.iloc[t]
            
            # Check open trade exits
            if active_trade:
                active_trade['hold_time'] += 1
                exit_price = 0
                reason = ""
                
                # Check Stop Loss
                if (active_trade['entry_signal'] == "BUY" and row['low'] <= active_trade['stop_loss']) or \
                   (active_trade['entry_signal'] == "SELL" and row['high'] >= active_trade['stop_loss']):
                    exit_price = active_trade['stop_loss']
                    reason = "Stop Loss"
                
                # Check Take Profit
                elif (active_trade['entry_signal'] == "BUY" and row['high'] >= active_trade['take_profit']) or \
                     (active_trade['entry_signal'] == "SELL" and row['low'] <= active_trade['take_profit']):
                    exit_price = active_trade['take_profit']
                    reason = "Take Profit"
                    
                # Time exit (16 candles = 4 hours)
                elif active_trade['hold_time'] >= 16:
                    exit_price = row['close']
                    reason = "Time Exit"
                    
                if exit_price > 0:
                    pos_size = active_trade['position_size']
                    entry_p = active_trade['entry_price']
                    
                    if active_trade['entry_signal'] == "BUY":
                        pnl = (exit_price - entry_p) * pos_size
                    else:
                        pnl = (entry_p - exit_price) * pos_size
                        
                    entry_fee = pos_size * entry_p * self.fee_pct
                    exit_fee = pos_size * exit_price * self.fee_pct
                    net_pnl = pnl - (entry_fee + exit_fee)
                    
                    capital += net_pnl
                    if capital > peak_capital:
                        peak_capital = capital
                    dd = (peak_capital - capital) / peak_capital * 100
                    if dd > max_drawdown:
                        max_drawdown = dd
                        
                    active_trade['exit_time'] = str(row['timestamp'])
                    active_trade['exit_price'] = float(exit_price)
                    active_trade['pnl_usd'] = float(net_pnl)
                    active_trade['pnl_pct'] = float((net_pnl / (pos_size * entry_p)) * 100)
                    active_trade['exit_reason'] = str(reason)
                    active_trade['win'] = bool(net_pnl > 0)
                    
                    # Also make sure entry stuff is float
                    active_trade['entry_price'] = float(active_trade['entry_price'])
                    active_trade['position_size'] = float(active_trade['position_size'])
                    active_trade['stop_loss'] = float(active_trade['stop_loss'])
                    active_trade['take_profit'] = float(active_trade['take_profit'])
                    
                    trades.append(active_trade)
                    active_trade = None
                    continue # Do not open another trade on the same tick we exit
            
            if not active_trade:
                # 1. Evaluate baseline signals
                # We mock the df passed to structure/volume/candles by passing df.iloc[:t+1]
                # To be efficient, we adapt the logic.
                votes = {}
                votes.update(_vote_trend(row))
                votes.update(_vote_momentum(row))
                votes.update(_vote_volatility(row))
                
                # Quick mock for volume, structure, candles that needed df
                # _vote_volume needs obv.iloc[-2]
                prev_obv = df.iloc[t-1]['obv'] if t > 0 else row['obv']
                votes['obv'] = 1 if row['obv'] > prev_obv else -1
                votes['vwap'] = 1 if row['close'] > row['vwap'] else -1
                votes['volume_spike'] = 1 if row['volume_spike'] and row['close'] > row['open'] else (-1 if row['volume_spike'] and row['close'] < row['open'] else 0)
                
                votes['support_resistance'] = 1 if (pd.notna(row['support']) and row['close'] <= row['support'] * 1.005) else (-1 if (pd.notna(row['resistance']) and row['close'] >= row['resistance'] * 0.995) else 0)
                # Trendline slope is in the row as trendline_slope? No, we didn't save it directly in features_engineered. Oh wait, we dropped raw support, etc.
                # Actually features_engineered has EVERYTHING before drop.
                
                # Candles are precomputed!
                votes['hammer'] = 1 if row.get('pattern_hammer', 0) else 0
                votes['shooting_star'] = -1 if row.get('pattern_shooting_star', 0) else 0
                votes['bullish_engulfing'] = 1 if row.get('pattern_bullish_engulfing', 0) else 0
                votes['bearish_engulfing'] = -1 if row.get('pattern_bearish_engulfing', 0) else 0
                votes['three_soldiers_crows'] = 1 if row.get('pattern_three_soldiers', 0) else 0
                
                bullish = sum(1 for v in votes.values() if v > 0)
                bearish = sum(1 for v in votes.values() if v < 0)
                
                threshold = 5 # Config.MIN_CONFIRMATIONS_TO_TRADE
                
                action = "HOLD"
                if bullish >= threshold and bullish > bearish:
                    action = "BUY"
                elif bearish >= threshold and bearish > bullish:
                    action = "SELL"
                    
                if action != "HOLD":
                    # Open position
                    atr = row['atr']
                    if pd.isna(atr) or atr == 0:
                        continue
                    
                    sl = calculate_stop_loss(row['close'], atr, 1.5, action)
                    tp = calculate_take_profit(row['close'], atr, 2.5, action)
                    sl_dist = abs(row['close'] - sl)
                    pos_size = calculate_position_size(capital, 0.015, sl_dist)
                    
                    # Ensure pos size doesn't exceed 3x leverage
                    max_pos = (capital * 3) / row['close']
                    pos_size = min(pos_size, max_pos)
                    
                    if pos_size > 0:
                        active_trade = {
                            'entry_time': str(row['timestamp']),
                            'entry_price': row['close'],
                            'entry_signal': action,
                            'position_size': pos_size,
                            'stop_loss': sl,
                            'take_profit': tp,
                            'hold_time': 0
                        }

        return trades, self.calculate_metrics(trades, capital, max_drawdown, df.iloc[0]['timestamp'], df.iloc[-1]['timestamp'], "Baseline")

    def calculate_metrics(self, trades, final_capital, max_drawdown, start_time, end_time, name):
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['win'])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        gross_profit = sum(t['pnl_usd'] for t in trades if t['pnl_usd'] > 0)
        gross_loss = abs(sum(t['pnl_usd'] for t in trades if t['pnl_usd'] < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Sharpe (simplified, assuming daily risk free rate 0)
        # Convert PnL % per trade to a series
        pnl_pcts = [t['pnl_pct'] for t in trades]
        if len(pnl_pcts) > 1:
            avg_ret = np.mean(pnl_pcts)
            std_ret = np.std(pnl_pcts)
            sharpe = (avg_ret / std_ret) if std_ret > 0 else 0
        else:
            sharpe = 0
            
        return {
            "name": name,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_return_pct": round(total_return, 2),
            "period": f"{start_time} to {end_time} (Test set)",
            "final_capital": round(final_capital, 2)
        }

    def execute_ml_predictions(self, df, y_pred, name="ML_Ensemble", sl_atr_mult=1.5, tp_atr_mult=2.5, time_exit_candles=4):
        """
        Backtest con predicciones de modelo ML.
        Abre trade solo cuando modelo predice BUY (1) o SELL (-1).
        Usa SL/TP/TimeExit igual que el Baseline, NO opera en HOLD (0).
        Parámetros sl_atr_mult y tp_atr_mult son configurables para optimización.
        """
        from backtest.risk_manager import calculate_position_size, calculate_stop_loss, calculate_take_profit

        capital = self.initial_capital
        trades = []
        active_trade = None
        peak_capital = capital
        max_drawdown = 0.0
        
        for t in range(len(df)):
            row = df.iloc[t]
            label = int(y_pred[t])
            
            # Check open trade exits
            if active_trade:
                active_trade['hold_time'] += 1
                exit_price = 0
                reason = ""
                
                if (active_trade['entry_signal'] == "BUY" and row['low'] <= active_trade['stop_loss']) or \
                   (active_trade['entry_signal'] == "SELL" and row['high'] >= active_trade['stop_loss']):
                    exit_price = active_trade['stop_loss']
                    reason = "Stop Loss"
                    
                elif (active_trade['entry_signal'] == "BUY" and row['high'] >= active_trade['take_profit']) or \
                     (active_trade['entry_signal'] == "SELL" and row['low'] <= active_trade['take_profit']):
                    exit_price = active_trade['take_profit']
                    reason = "Take Profit"
                    
                elif active_trade['hold_time'] >= time_exit_candles:
                    exit_price = row['close']
                    reason = "Time Exit"
                    
                if exit_price > 0:
                    pos_size = active_trade['position_size']
                    entry_p = active_trade['entry_price']
                    
                    if active_trade['entry_signal'] == "BUY":
                        pnl = (exit_price - entry_p) * pos_size
                    else:
                        pnl = (entry_p - exit_price) * pos_size
                        
                    net_pnl = pnl - (pos_size * entry_p + pos_size * exit_price) * self.fee_pct
                    capital += net_pnl
                    
                    if capital > peak_capital:
                        peak_capital = capital
                    dd = (peak_capital - capital) / peak_capital * 100
                    if dd > max_drawdown:
                        max_drawdown = dd
                    
                    active_trade['exit_time'] = str(row['timestamp'])
                    active_trade['exit_price'] = float(exit_price)
                    active_trade['pnl_usd'] = float(net_pnl)
                    active_trade['pnl_pct'] = float((net_pnl / (pos_size * entry_p)) * 100)
                    active_trade['exit_reason'] = reason
                    active_trade['win'] = bool(net_pnl > 0)
                    active_trade['entry_price'] = float(active_trade['entry_price'])
                    active_trade['position_size'] = float(active_trade['position_size'])
                    active_trade['stop_loss'] = float(active_trade['stop_loss'])
                    active_trade['take_profit'] = float(active_trade['take_profit'])
                    trades.append(active_trade)
                    active_trade = None
                    
            if not active_trade and label != 0:
                action = "BUY" if label == 1 else "SELL"
                atr = row.get('atr', 0)
                if pd.isna(atr) or atr <= 0:
                    continue
                    
                sl = calculate_stop_loss(row['close'], atr, sl_atr_mult, action)
                tp = calculate_take_profit(row['close'], atr, tp_atr_mult, action)
                sl_dist = abs(row['close'] - sl)
                pos_size = calculate_position_size(capital, 0.015, sl_dist)
                max_pos = (capital * 3) / row['close']
                pos_size = min(pos_size, max_pos)
                
                if pos_size > 0:
                    active_trade = {
                        'entry_time': str(row['timestamp']),
                        'entry_price': row['close'],
                        'entry_signal': action,
                        'position_size': pos_size,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'hold_time': 0
                    }
                    
        return trades, self.calculate_metrics(trades, capital, max_drawdown, df.iloc[0]['timestamp'], df.iloc[-1]['timestamp'], name)
