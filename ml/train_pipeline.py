import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.model_trainer import ModelTrainer
from ml.model_evaluator import ModelEvaluator
from ml.model_inference import ModelInference

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, 'data', 'processed')
outputs_dir = os.path.join(base_dir, 'outputs')
models_dir = os.path.join(base_dir, 'models')

def load_data():
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"Data loaded. X_train shape: {X_train.shape}")
    
    # 1. Train models
    trainer = ModelTrainer()
    trainer.train_xgboost(X_train, X_val, y_train, y_val)
    trainer.train_lightgbm(X_train, X_val, y_train, y_val)
    
    # 2. Inference
    inference = ModelInference(models_dir)
    y_pred_ensemble, y_proba_ensemble, xgb_proba, lgb_proba = inference.batch_predict(X_test, confidence_threshold=0.45)
    
    xgb_max = np.max(xgb_proba, axis=1)
    y_pred_xgb = np.argmax(xgb_proba, axis=1)
    y_pred_xgb[xgb_max < 0.45] = 1
    y_pred_xgb = inference._unmap_labels(y_pred_xgb)
    
    lgb_max = np.max(lgb_proba, axis=1)
    y_pred_lgb = np.argmax(lgb_proba, axis=1)
    y_pred_lgb[lgb_max < 0.45] = 1
    y_pred_lgb = inference._unmap_labels(y_pred_lgb)
    
    # 3. Evaluate ML Metrics
    evaluator = ModelEvaluator()
    xgb_metrics = evaluator.evaluate_ml_metrics(y_test, y_pred_xgb, xgb_proba)
    lgb_metrics = evaluator.evaluate_ml_metrics(y_test, y_pred_lgb, lgb_proba)
    ens_metrics = evaluator.evaluate_ml_metrics(y_test, y_pred_ensemble, y_proba_ensemble)
    
    # 4. Evaluate Trading Performance
    # Load df_test and drop NaNs exactly like we did in feature_engineering
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
    
    # Actually backtest baseline vs models
    from backtest.backtest_engine import BacktestEngine
    engine = BacktestEngine(initial_capital=15.0)
    baseline_trades, baseline_perf = engine.execute_baseline(df_test)
    
    _, xgb_perf = evaluator.evaluate_trading_performance(y_pred_xgb, df_test, "XGBoost")
    _, lgb_perf = evaluator.evaluate_trading_performance(y_pred_lgb, df_test, "LightGBM")
    ens_trades, ens_perf = evaluator.evaluate_trading_performance(y_pred_ensemble, df_test, "Ensemble")
    
    # Save training report
    report = {
        "xgboost_ml": xgb_metrics,
        "lightgbm_ml": lgb_metrics,
        "ensemble_ml": ens_metrics,
        "trading_performance": {
            "baseline": baseline_perf,
            "xgboost": xgb_perf,
            "lightgbm": lgb_perf,
            "ensemble": ens_perf
        }
    }
    
    with open(os.path.join(outputs_dir, 'ml_training_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n=== ML METRICS ON TEST SET ===")
    print(f"{'Model':<15} | {'Accuracy':<10} | {'F1 Weighted':<15} | {'ROC-AUC':<10}")
    print("-" * 55)
    print(f"{'XGBoost':<15} | {xgb_metrics['accuracy']:<10.4f} | {xgb_metrics['f1_weighted']:<15.4f} | {xgb_metrics['roc_auc']:<10.4f}")
    print(f"{'LightGBM':<15} | {lgb_metrics['accuracy']:<10.4f} | {lgb_metrics['f1_weighted']:<15.4f} | {lgb_metrics['roc_auc']:<10.4f}")
    print(f"{'Ensemble':<15} | {ens_metrics['accuracy']:<10.4f} | {ens_metrics['f1_weighted']:<15.4f} | {ens_metrics['roc_auc']:<10.4f}")
    
    print("\n=== TRADING METRICS ON TEST SET ===")
    print(f"{'Model':<15} | {'Win Rate %':<10} | {'Profit Factor':<15} | {'Total Return %':<15}")
    print("-" * 65)
    print(f"{'Baseline':<15} | {baseline_perf['win_rate_pct']:<10.2f} | {baseline_perf['profit_factor']:<15.2f} | {baseline_perf['total_return_pct']:<15.2f}")
    print(f"{'XGBoost':<15} | {xgb_perf['win_rate_pct']:<10.2f} | {xgb_perf['profit_factor']:<15.2f} | {xgb_perf['total_return_pct']:<15.2f}")
    print(f"{'LightGBM':<15} | {lgb_perf['win_rate_pct']:<10.2f} | {lgb_perf['profit_factor']:<15.2f} | {lgb_perf['total_return_pct']:<15.2f}")
    print(f"{'Ensemble':<15} | {ens_perf['win_rate_pct']:<10.2f} | {ens_perf['profit_factor']:<15.2f} | {ens_perf['total_return_pct']:<15.2f}")
    print(f"Report saved to {os.path.join(outputs_dir, 'ml_training_report.json')}")
    
if __name__ == "__main__":
    main()
