import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

print("Imports completed...")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.model_inference import ModelInference
from ml.gemini_validator_v2 import GeminiValidatorV2
from backtest.backtest_engine import BacktestEngine
from backtest.risk_manager import calculate_position_size, calculate_stop_loss, calculate_take_profit

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, 'data', 'processed')
outputs_dir = os.path.join(base_dir, 'outputs')
models_dir = os.path.join(base_dir, 'models')

def main():
    print("Loading data for Gemini Backtest...")
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    
    # Load raw features for context and prices
    df_eng = pd.read_csv(os.path.join(processed_dir, 'features_engineered.csv'))
    cols_to_drop = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'sma_50', 'sma_200', 'ema_20', 'support', 'resistance',
        'vwap', 'bb_mid', 'bb_upper', 'bb_lower',
        'returns_2_candles', 'LABEL'
    ]
    features_cols = [c for c in df_eng.columns if c not in cols_to_drop]
    df_clean = df_eng.dropna(subset=features_cols + ['LABEL']).reset_index(drop=True)
    val_end = int(len(df_clean) * 0.9)
    df_test = df_clean.iloc[val_end:].copy().reset_index(drop=True)

    print("Loading ML models...")
    inference = ModelInference(models_dir)
    # Get raw predictions without threshold filtering
    y_pred_ml, y_proba_ml, _, _ = inference.batch_predict(X_test, confidence_threshold=0.0)
    
    print("Initializing Gemini Validator v2...")
    validator = GeminiValidatorV2()
    if not validator.client:
        print("WARNING: GEMINI_API_KEY no encontrada en .env. Usando Mock Validator.")
        
    engine = BacktestEngine(initial_capital=15.0)
    
    validation_log = []
    
    # We'll run the backtest loop manually to inject Gemini
    capital = engine.initial_capital
    trades = []
    active_trade = None
    peak_capital = capital
    max_drawdown = 0.0
    
    print(f"Starting Gemini Backtest over {len(df_test)} candles...")
    
    # Phase 6.5 — new SL/TP from Phase 5.5 analysis and config.py
    sl_atr_mult       = 0.8    # down from 1.25 — ratio 1:3.75
    tp_atr_mult       = 3.0    # up from 1.50
    time_exit_candles = 4
    ml_conf_threshold = 0.35   # down from 0.40 — let more signals through
    
    for t in range(len(df_test)):
        row = df_test.iloc[t]
        label = int(y_pred_ml[t])
        confidence = float(np.max(y_proba_ml[t]))
        
        # 1. Check open trade exits (same logic as execute_ml_predictions)
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
                    
                net_pnl = pnl - (pos_size * entry_p + pos_size * exit_price) * engine.fee_pct
                capital += net_pnl
                
                if capital > peak_capital: peak_capital = capital
                dd = (peak_capital - capital) / peak_capital * 100
                if dd > max_drawdown: max_drawdown = dd
                
                active_trade['exit_time'] = str(row['timestamp'])
                active_trade['exit_price'] = float(exit_price)
                active_trade['pnl_usd'] = float(net_pnl)
                active_trade['pnl_pct'] = float((net_pnl / (pos_size * entry_p)) * 100)
                active_trade['exit_reason'] = reason
                active_trade['win'] = bool(net_pnl > 0)
                trades.append(active_trade)
                active_trade = None

        # 2. Check for new signals
        if not active_trade and label != 0 and confidence >= ml_conf_threshold:
            pred_action = "BUY" if label == 1 else "SELL"
            
            # Scorer local (backtest) — replica las reglas del prompt sin llamar a la API
            market_context = validator.prepare_market_context(row)
            gemini_result  = validator.validate_local(pred_action, confidence, market_context)
            score    = gemini_result.get('gemini_score', 0)
            decision = gemini_result.get('gemini_decision', 'SKIP')
            
            should_exec = validator.should_trade(
                gemini_score=score,
                ml_confidence=confidence
            )
            
            log_entry = {
                'timestamp': str(row['timestamp']),
                'close': float(row['close']),
                'ml_prediction': pred_action,
                'ml_confidence': float(confidence),
                'gemini_score': gemini_result.get('gemini_score', 0),
                'gemini_decision': gemini_result.get('gemini_decision', 'SKIP'),
                'gemini_reasoning': gemini_result.get('gemini_reasoning', ''),
                'actually_traded': should_exec
            }
            
            if should_exec:
                atr = row.get('atr', 0)
                if not pd.isna(atr) and atr > 0:
                    sl = calculate_stop_loss(row['close'], atr, sl_atr_mult, pred_action)
                    tp = calculate_take_profit(row['close'], atr, tp_atr_mult, pred_action)
                    sl_dist = abs(row['close'] - sl)
                    pos_size = calculate_position_size(capital, 0.015, sl_dist)
                    max_pos = (capital * 3) / row['close']
                    pos_size = min(pos_size, max_pos)
                    
                    if pos_size > 0:
                        active_trade = {
                            'entry_time': str(row['timestamp']),
                            'entry_price': row['close'],
                            'entry_signal': pred_action,
                            'position_size': pos_size,
                            'stop_loss': sl,
                            'take_profit': tp,
                            'hold_time': 0,
                            'log_ref': log_entry # Reference to update pnl later
                        }
            else:
                log_entry['reason'] = gemini_result.get('rejection_reasons', ['Skipped by Gemini'])
                
            validation_log.append(log_entry)

    # Update log entries with PnL for executed trades
    for t in trades:
        if 'log_ref' in t:
            t['log_ref']['pnl'] = t['pnl_usd']
            t['log_ref']['win'] = t['win']
            del t['log_ref'] # Clean up

    # 3. Calculate metrics
    ml_gemini_metrics = engine.calculate_metrics(
        trades, capital, max_drawdown, 
        df_test.iloc[0]['timestamp'], df_test.iloc[-1]['timestamp'], "ML + Gemini"
    )
    
    # 4. Compare with ML Only
    print("\nRunning ML Only for comparison...")
    # Apply threshold 0.45 like we did in phase 5
    y_pred_filtered, _, _, _ = inference.batch_predict(X_test, confidence_threshold=0.45)
    ml_only_trades, ml_only_metrics = engine.execute_ml_predictions(
        df_test, y_pred_filtered, name="ML Only", sl_atr_mult=1.25, tp_atr_mult=1.50
    )
    
    # 5. Baseline
    print("Running Baseline for comparison...")
    baseline_trades, baseline_metrics = engine.execute_baseline(df_test)
    
    # 6. Save Reports
    print("\nGenerating Reports...")
    skipped = sum(1 for x in validation_log if not x['actually_traded'])
    total_opps = len(validation_log)
    
    report = {
        "baseline": baseline_metrics,
        "ml_only": ml_only_metrics,
        "ml_plus_gemini": ml_gemini_metrics,
        "comparison": {
            "trades_baseline": baseline_metrics['total_trades'],
            "trades_ml": ml_only_metrics['total_trades'],
            "trades_ml_gemini": ml_gemini_metrics['total_trades'],
            "gemini_skip_rate": round(skipped / total_opps * 100, 1) if total_opps > 0 else 0,
            "gemini_effectiveness": {
                "pf_improvement": round(ml_gemini_metrics['profit_factor'] - ml_only_metrics['profit_factor'], 2),
                "return_improvement": round(ml_gemini_metrics['total_return_pct'] - ml_only_metrics['total_return_pct'], 2)
            }
        }
    }
    
    with open(os.path.join(outputs_dir, 'gemini_backtest_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    with open(os.path.join(outputs_dir, 'gemini_validation_log.json'), 'w') as f:
        json.dump(validation_log, f, indent=4)
        
    print("\n" + "="*60)
    print(" GEMINI INTEGRATION RESULTS (TEST SET)")
    print("="*60)
    print(f"Opportunities Found (Conf > 0.40): {total_opps}")
    print(f"Skipped by Gemini: {skipped} ({(skipped/total_opps*100) if total_opps else 0:.1f}%)")
    print("\nPERFORMANCE COMPARISON:")
    print(f"{'Metric':<15} | {'Baseline':<10} | {'ML Only':<10} | {'ML+Gemini':<10}")
    print("-" * 55)
    print(f"{'Win Rate':<15} | {baseline_metrics['win_rate_pct']:>9.1f}% | {ml_only_metrics['win_rate_pct']:>9.1f}% | {ml_gemini_metrics['win_rate_pct']:>9.1f}%")
    print(f"{'Profit Factor':<15} | {baseline_metrics['profit_factor']:>10.2f} | {ml_only_metrics['profit_factor']:>10.2f} | {ml_gemini_metrics['profit_factor']:>10.2f}")
    print(f"{'Total Return':<15} | {baseline_metrics['total_return_pct']:>9.1f}% | {ml_only_metrics['total_return_pct']:>9.1f}% | {ml_gemini_metrics['total_return_pct']:>9.1f}%")
    print(f"{'Trades Executed':<15} | {baseline_metrics['total_trades']:>10} | {ml_only_metrics['total_trades']:>10} | {ml_gemini_metrics['total_trades']:>10}")
    print("="*60)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
