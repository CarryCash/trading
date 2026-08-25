import os
import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb

class ModelInference:
    def __init__(self, models_dir):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(models_dir, 'xgboost_model.json'))
        self.lgb_model = joblib.load(os.path.join(models_dir, 'lightgbm_model.pkl'))
        
    def _unmap_labels(self, y):
        # Maps 0, 1, 2 to -1, 0, 1
        return y - 1
        
    def batch_predict(self, X, confidence_threshold=0.0):
        xgb_proba = self.xgb_model.predict_proba(X)
        lgb_proba = self.lgb_model.predict_proba(X)
        
        ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
        
        max_probas = np.max(ensemble_proba, axis=1)
        y_pred = np.argmax(ensemble_proba, axis=1)
        
        # If below threshold, set prediction to 1 (which maps to 0 / HOLD)
        if confidence_threshold > 0:
            y_pred[max_probas < confidence_threshold] = 1
            
        y_pred_unmapped = self._unmap_labels(y_pred)
        
        return y_pred_unmapped, ensemble_proba, xgb_proba, lgb_proba
