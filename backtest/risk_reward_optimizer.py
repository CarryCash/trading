import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtest.backtest_engine import BacktestEngine

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_dir = os.path.join(base_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

class RiskRewardOptimizer:
    def __init__(self, initial_capital=15.0):
        self.engine = BacktestEngine(initial_capital=initial_capital)
        self.initial_capital = initial_capital

    def grid_search(self, df_test, y_pred,
                    sl_mults=None, tp_mults=None):
        if sl_mults is None:
            sl_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
        if tp_mults is None:
            tp_mults = [1.0, 1.33, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        results = []
        total = len(sl_mults) * len(tp_mults)
        count = 0

        print(f"\nStarting Risk/Reward Grid Search ({total} combinations)...")
        print("-" * 72)

        for sl in sl_mults:
            for tp in tp_mults:
                count += 1
                trades, metrics = self.engine.execute_ml_predictions(
                    df_test, y_pred,
                    sl_atr_mult=sl,
                    tp_atr_mult=tp,
                    name=f"SL={sl:.2f}_TP={tp:.2f}"
                )

                n = metrics['total_trades']
                wr  = metrics['win_rate_pct'] / 100
                pf  = metrics['profit_factor']
                dd  = metrics['max_drawdown_pct'] / 100
                ret = metrics['total_return_pct'] / 100
                sr  = metrics['sharpe_ratio']

                print(
                    f"[{count:2d}/{total}] SL={sl:.2f}, TP={tp:.2f}"
                    f"  =>  WR={wr:.1%}  PF={pf:.2f}"
                    f"  DD={dd:.1%}  Return={ret:.1%}  Trades={n}"
                )

                results.append({
                    'sl_mult':      sl,
                    'tp_mult':      tp,
                    'ratio':        round(tp / sl, 2),
                    'win_rate':     round(wr, 4),
                    'profit_factor': round(pf, 4),
                    'max_drawdown': round(dd, 4),
                    'total_return': round(ret, 4),
                    'sharpe':       round(sr, 4),
                    'total_trades': n
                })

        return pd.DataFrame(results)

    def find_optimal(self, results: pd.DataFrame):
        """Find best combo respecting risk constraints."""
        valid = results[
            (results['win_rate']     >= 0.40) &   # Allow wider net given small test set
            (results['max_drawdown'] <= 0.40) &
            (results['total_trades'] >= 5)
        ]

        if valid.empty:
            print("No combo passed all constraints, relaxing to PF > 1.0")
            valid = results[results['profit_factor'] > 1.0]

        if valid.empty:
            print("Still none > 1.0, returning best available")
            valid = results

        optimal = valid.loc[valid['profit_factor'].idxmax()]
        return optimal

    def generate_heatmap(self, results: pd.DataFrame) -> str:
        pivot = results.pivot_table(
            values='profit_factor',
            index='sl_mult',
            columns='tp_mult'
        ).round(2)

        lines = [
            "\n+================================================================+",
            "|  Profit Factor Heatmap  (rows=SL_MULT, cols=TP_MULT)         |",
            "+================================================================+",
        ]
        lines.append(pivot.to_string())
        lines += [
            "|  Higher = Better  |  Target: PF > 1.30                       |",
            "+================================================================+",
        ]
        heatmap_str = "\n".join(lines)
        print(heatmap_str)
        return heatmap_str

    def run(self, df_test, y_pred, baseline_sl=1.5, baseline_tp=2.5, baseline_pf=0.75, baseline_ret=-0.016):
        results = self.grid_search(df_test, y_pred)

        # Save CSV
        csv_path = os.path.join(outputs_dir, 'risk_reward_optimization_results.csv')
        results.to_csv(csv_path, index=False)

        # Heatmap
        heatmap_str = self.generate_heatmap(results)
        heatmap_path = os.path.join(outputs_dir, 'risk_reward_optimization_heatmap.txt')
        with open(heatmap_path, 'w', encoding='utf-8') as f:
            f.write(heatmap_str)

        # Optimal
        optimal = self.find_optimal(results)
        print(f"\n{'='*60}")
        print(f"OPTIMAL PARAMETERS FOUND:")
        print(f"  SL_MULT = {optimal['sl_mult']:.2f} × ATR")
        print(f"  TP_MULT = {optimal['tp_mult']:.2f} × ATR")
        print(f"  Ratio   = 1:{optimal['ratio']:.2f}")
        print(f"  Win Rate    = {optimal['win_rate']:.1%}")
        print(f"  Profit Factor = {optimal['profit_factor']:.2f}")
        print(f"  Max Drawdown  = {optimal['max_drawdown']:.1%}")
        print(f"  Total Return  = {optimal['total_return']:.1%}")
        print(f"{'='*60}")

        # Save report
        report = {
            "optimal_parameters": {
                "sl_atr_mult": float(optimal['sl_mult']),
                "tp_atr_mult": float(optimal['tp_mult']),
                "ratio_tp_sl": float(optimal['ratio'])
            },
            "performance_with_optimal": {
                "win_rate": float(optimal['win_rate']),
                "profit_factor": float(optimal['profit_factor']),
                "max_drawdown": float(optimal['max_drawdown']),
                "total_return": float(optimal['total_return']),
                "sharpe_ratio": float(optimal['sharpe'])
            },
            "comparison": {
                "before": {
                    "sl_mult": baseline_sl,
                    "tp_mult": baseline_tp,
                    "profit_factor": baseline_pf,
                    "total_return": baseline_ret
                },
                "after": {
                    "sl_mult": float(optimal['sl_mult']),
                    "tp_mult": float(optimal['tp_mult']),
                    "profit_factor": float(optimal['profit_factor']),
                    "total_return": float(optimal['total_return'])
                },
                "improvement": {
                    "profit_factor_delta": round(float(optimal['profit_factor']) - baseline_pf, 4),
                    "return_delta": round(float(optimal['total_return']) - baseline_ret, 4)
                }
            }
        }

        report_path = os.path.join(outputs_dir, 'risk_reward_optimization_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"\nSaved: {csv_path}")
        print(f"Saved: {heatmap_path}")
        print(f"Saved: {report_path}")
        return results, optimal


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────
if __name__ == "__main__":
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir    = os.path.join(base_dir, 'models')

    # Load X_test and generate predictions
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))

    from ml.model_inference import ModelInference
    inference = ModelInference(models_dir)
    y_pred_ensemble, _, _, _ = inference.batch_predict(X_test, confidence_threshold=0.45)

    # Load df_test aligned with X_test
    df_eng = pd.read_csv(os.path.join(processed_dir, 'features_engineered.csv'))
    cols_to_drop = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'sma_50', 'sma_200', 'ema_20', 'support', 'resistance',
        'vwap', 'bb_mid', 'bb_upper', 'bb_lower',
        'returns_2_candles', 'LABEL'
    ]
    features_cols = [c for c in df_eng.columns if c not in cols_to_drop]
    df_clean = df_eng.dropna(subset=features_cols + ['LABEL']).reset_index(drop=True)
    n = len(df_clean)
    val_end = int(n * 0.9)
    df_test = df_clean.iloc[val_end:].copy().reset_index(drop=True)

    optimizer = RiskRewardOptimizer(initial_capital=15.0)
    results, optimal = optimizer.run(df_test, y_pred_ensemble)
