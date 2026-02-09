import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import time

INPUT_PATH = "data/processed/train_merged.csv"
TARGET_COLS = ['total alkalinity', 'electrical conductance', 'dissolved reactive phosphorus']
DROP_COLS = ['date', 'latitude', 'longitude', 'id']

def objective(trial):
    df = pd.read_csv(INPUT_PATH)
    leakage_cols = [c for c in df.columns if c.endswith('_log')]
    X = df.drop(columns=TARGET_COLS + DROP_COLS + leakage_cols, errors='ignore')
    y = df[TARGET_COLS]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 튜닝할 파라미터 범주 정의
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    scores = []
    for target in TARGET_COLS:
        y_tr_log = np.log1p(np.maximum(y_train[target], 0))
        y_va_log = np.log1p(np.maximum(y_val[target], 0))
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_train, y_tr_log, eval_set=[(X_val, y_va_log)], verbose=False)
        
        preds = np.expm1(model.predict(X_val))
        scores.append(r2_score(y_val[target], preds))
    
    # $$ Mean R^2 = \frac{1}{3} \sum_{i=1}^{3} R^2_i $$_
    return np.mean(scores)

def run_tuning():
    print(">> 하이퍼파라미터 최적화 시작 (Trials: 20)...")
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("\n>> 최적 파라미터:")
    print(study.best_params)
    print(f"최고 평균 R2: {study.best_value:.4f}")
    
    # 최적 파라미터 저장
    with open("best_params.pkl", "wb") as f:
        pickle.dump(study.best_params, f)
        
    elapsed = time.time() - start_time
    print(f">> 튜닝 종료. 소요 시간: {elapsed:.2f}s")

if __name__ == "__main__":
    run_tuning()
