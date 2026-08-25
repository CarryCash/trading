import os
import sys
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None

    def _map_labels(self, y):
        # Maps -1, 0, 1 to 0, 1, 2
        return y + 1

    def train_xgboost(self, X_train, X_val, y_train, y_val):
        y_t = self._map_labels(y_train)
        y_v = self._map_labels(y_val)
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=0.5,
            min_child_weight=5,
            gamma=0.1,
            early_stopping_rounds=50,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        )
        
        print("\n=== Training XGBoost ===")
        self.xgb_model.fit(
            X_train, y_t,
            eval_set=[(X_train, y_t), (X_val, y_v)],
            verbose=50
        )
        
        self.xgb_model.save_model(os.path.join(models_dir, 'xgboost_model.json'))
        return self.xgb_model

    def train_lightgbm(self, X_train, X_val, y_train, y_val):
        y_t = self._map_labels(y_train)
        y_v = self._map_labels(y_val)
        
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=0.5,
            num_leaves=25,
            min_child_samples=20,
            bagging_freq=5,
            objective='multiclass',
            num_class=3,
            random_state=42,
            verbose=-1
        )
        
        print("\n=== Training LightGBM ===")
        self.lgb_model.fit(
            X_train, y_t,
            eval_set=[(X_train, y_t), (X_val, y_v)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)]
        )
        
        joblib.dump(self.lgb_model, os.path.join(models_dir, 'lightgbm_model.pkl'))
        return self.lgb_model
