import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import time
import os

INPUT_PATH = "data/processed/train_merged.csv"
MODEL_PATH = "xgb_models_final.pkl"
TARGET_COLS = ['total alkalinity', 'electrical conductance', 'dissolved reactive phosphorus']
DROP_COLS = ['date', 'latitude', 'longitude', 'id']

def train():
    df = pd.read_csv(INPUT_PATH)
    leakage_cols = [c for c in df.columns if c.endswith('_log')]
    X = df.drop(columns=TARGET_COLS + DROP_COLS + leakage_cols, errors='ignore')
    y = df[TARGET_COLS]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optuna 최적 파라미터 적용
    best_params = {
        'n_estimators': 1977,
        'max_depth': 9,
        'learning_rate': 0.020588579077031266,
        'subsample': 0.905886633301085,
        'colsample_bytree': 0.6068424529868086,
        'min_child_weight': 1,
        'n_jobs': -1,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    models = {}
    start_time = time.time()
    print(f">> 최적 파라미터로 최종 학습 시작. 피처 수: {X.shape[1]}")

    for i, target in enumerate(TARGET_COLS):
        y_tr_log = np.log1p(np.maximum(y_train[target], 0))
        y_va_log = np.log1p(np.maximum(y_val[target], 0))
        
        model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
        model.fit(X_train, y_tr_log, eval_set=[(X_val, y_va_log)], verbose=False)
        
        preds = np.expm1(model.predict(X_val))
        r2 = r2_score(y_val[target], preds)
        models[target] = model
        
        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        remaining = avg * (len(TARGET_COLS) - (i + 1))
        print(f"[{target}] R2: {r2:.4f} | 남은 시간: {remaining:.2f}s")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({'models': models, 'features': X.columns.tolist()}, f)
    print(f">> 최종 모델 저장 완료. 총 소요: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train()
