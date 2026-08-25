import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.backtest_engine import BacktestEngine
from backtest.risk_manager import calculate_position_size

def test_risk_manager():
    capital = 15.0
    # 1.5% risk = $0.225
    # SL distance = $1000
    pos_size = calculate_position_size(capital, 0.015, 1000)
    assert pos_size == pytest.approx(0.000225)
    
    # 0 distance should return 0
    assert calculate_position_size(capital, 0.01, 0) == 0

def test_metrics_calculation():
    engine = BacktestEngine(15.0)
    trades = [
        {'win': True, 'pnl_usd': 5.0, 'pnl_pct': 10},
        {'win': False, 'pnl_usd': -2.0, 'pnl_pct': -5},
        {'win': True, 'pnl_usd': 3.0, 'pnl_pct': 8}
    ]
    # Gross profit: 8, Gross loss: 2 -> PF: 4
    metrics = engine.calculate_metrics(trades, 21.0, 10.0, "2026", "2026", "Mock")
    assert metrics['win_rate_pct'] == 66.67
    assert metrics['profit_factor'] == 4.0
    assert metrics['total_return_pct'] == 40.0 # (21-15)/15 = 40%

def test_oracle_backtesting():
    engine = BacktestEngine(15.0)
    # Mock DF
    data = []
    prices = [100, 110, 120, 115, 105, 95]
    labels = [1, 0, -1, 0, 0, 0] # BUY at 100 (exit at 120), SELL at 120 (exit at 105)
    
    for i in range(len(prices)):
        data.append({
            'timestamp': f"2026-01-01 0{i}:00",
            'close': prices[i],
            'LABEL': labels[i]
        })
        
    df = pd.DataFrame(data)
    trades, metrics = engine.execute_oracle(df)
    
    assert len(trades) == 2
    assert trades[0]['entry_signal'] == "BUY"
    assert trades[0]['pnl_usd'] > 0
    assert trades[1]['entry_signal'] == "SELL"
    assert trades[1]['pnl_usd'] > 0
    assert metrics['win_rate_pct'] == 100.0
