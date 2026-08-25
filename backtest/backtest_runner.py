import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.backtest_engine import BacktestEngine

def run_backtest():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load engineered features (contains ALL columns)
    df = pd.read_csv(os.path.join(processed_dir, 'features_engineered.csv'))
    
    # We want to run backtest ONLY on the test set.
    # The split in phase 3 was last 10%.
    n = len(df)
    val_end = int(n * 0.9)
    df_test = df.iloc[val_end:].copy().reset_index(drop=True)
    
    engine = BacktestEngine(initial_capital=15.0)
    
    # Baseline
    print("Executing Baseline Backtest...")
    baseline_trades, baseline_metrics = engine.execute_baseline(df_test)
    
    # Oracle
    print("Executing Oracle Backtest...")
    oracle_trades, oracle_metrics = engine.execute_oracle(df_test)
    
    # Save reports
    baseline_report = {
        "trades": baseline_trades,
        "summary": baseline_metrics
    }
    with open(os.path.join(outputs_dir, 'backtest_baseline_report.json'), 'w') as f:
        json.dump(baseline_report, f, indent=4)
        
    oracle_report = {
        "trades": oracle_trades,
        "summary": oracle_metrics
    }
    with open(os.path.join(outputs_dir, 'backtest_oracle_report.json'), 'w') as f:
        json.dump(oracle_report, f, indent=4)
        
    # Print comparison
    print("\n=== BACKTESTING COMPARISON ===")
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Oracle':<10}")
    print("-" * 45)
    for k in baseline_metrics.keys():
        if k not in ['name', 'period']:
            print(f"{k:<20} | {baseline_metrics[k]:<10} | {oracle_metrics[k]:<10}")
            
    # Generate HTML
    html_content = f"""
    <html>
    <head><title>Backtest Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
    </head>
    <body>
        <h2>Metrics Comparison</h2>
        <table>
            <tr><th>Metric</th><th>Baseline</th><th>Oracle</th></tr>
            <tr><td>Total Trades</td><td>{baseline_metrics['total_trades']}</td><td>{oracle_metrics['total_trades']}</td></tr>
            <tr><td>Win Rate %</td><td>{baseline_metrics['win_rate_pct']}</td><td>{oracle_metrics['win_rate_pct']}</td></tr>
            <tr><td>Profit Factor</td><td>{baseline_metrics['profit_factor']}</td><td>{oracle_metrics['profit_factor']}</td></tr>
            <tr><td>Max Drawdown %</td><td>{baseline_metrics['max_drawdown_pct']}</td><td>{oracle_metrics['max_drawdown_pct']}</td></tr>
            <tr><td>Sharpe Ratio</td><td>{baseline_metrics['sharpe_ratio']}</td><td>{oracle_metrics['sharpe_ratio']}</td></tr>
            <tr><td>Total Return %</td><td>{baseline_metrics['total_return_pct']}</td><td>{oracle_metrics['total_return_pct']}</td></tr>
        </table>
    </body>
    </html>
    """
    with open(os.path.join(outputs_dir, 'backtest_comparison.html'), 'w') as f:
        f.write(html_content)
        
    print(f"\nSaved reports to {outputs_dir}/")

if __name__ == "__main__":
    run_backtest()
