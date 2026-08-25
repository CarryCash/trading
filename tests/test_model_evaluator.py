import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.model_evaluator import ModelEvaluator

def test_ml_metrics():
    evaluator = ModelEvaluator()
    y_true = np.array([-1, 0, 1, 0])
    y_pred = np.array([-1, 1, 1, 0]) # 3 correct out of 4 (75% accuracy)
    
    # Mock probas
    y_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.3, 0.6], # Error here
        [0.1, 0.1, 0.8],
        [0.1, 0.7, 0.2]
    ])
    
    metrics = evaluator.evaluate_ml_metrics(y_true, y_pred, y_proba)
    assert metrics['accuracy'] == 0.75
    assert len(metrics['confusion_matrix']) == 3
