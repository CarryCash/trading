import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtest.backtest_engine import BacktestEngine

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_ml_metrics(self, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        try:
            roc = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError:
            roc = 0.5
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        
        return {
            'accuracy': float(acc),
            'f1_weighted': float(f1),
            'roc_auc': float(roc),
            'confusion_matrix': cm.tolist()
        }
        
    def evaluate_trading_performance(self, y_pred, df_test, name="ML_Model"):
        engine = BacktestEngine(initial_capital=15.0)
        # Use the ML-specific method that respects HOLD signals and uses SL/TP
        trades, metrics = engine.execute_ml_predictions(df_test, y_pred, name=name)
        return trades, metrics
