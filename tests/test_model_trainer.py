import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.model_trainer import ModelTrainer

def test_model_trainer_initialization():
    trainer = ModelTrainer()
    assert trainer.xgb_model is None
    assert trainer.lgb_model is None

def test_map_labels():
    trainer = ModelTrainer()
    y = np.array([-1, 0, 1])
    mapped = trainer._map_labels(y)
    assert np.array_equal(mapped, [0, 1, 2])
