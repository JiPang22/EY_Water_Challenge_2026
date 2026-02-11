import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pickle
import os

INPUT_PATH = "data/processed/train_merged.csv"
MODEL_PATH = "ensemble_models.pkl"
TARGET_COLS = ['total alkalinity', 'electrical conductance', 'dissolved reactive phosphorus']
DROP_COLS = ['date', 'latitude', 'longitude', 'id']

def train_cv():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} 파일이 없습니다.")
        return

    df = pd.read_csv(INPUT_PATH)
    leakage_cols = [c for c in df.columns if c.endswith('_log')]
    X = df.drop(columns=TARGET_COLS + DROP_COLS + leakage_cols, errors='ignore')
    y = df[TARGET_COLS]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_models = {target: [] for target in TARGET_COLS}
    
    # 모델 파라미터 설정
    xgb_params = {
        'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.05,
        'n_jobs': -1, 'tree_method': 'hist', 'random_state': 42, 'early_stopping_rounds': 50
    }
    
    lgb_params = {
        'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31,
        'n_jobs': -1, 'random_state': 42
    }

    for target in TARGET_COLS:
        print(f">> [{target}] 학습 시작...")
        y_log = np.log1p(np.maximum(y[target], 0))
        
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]
            
            # XGBoost
            m_xgb = xgb.XGBRegressor(**xgb_params)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            # LightGBM
            m_lgb = lgb.LGBMRegressor(**lgb_params)
            m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], 
                      eval_metric='l2', callbacks=[lgb.early_stopping(50, verbose=False)])
            
            p_xgb = m_xgb.predict(X_va)
            p_lgb = m_lgb.predict(X_va)
            p_ens = (p_xgb * 0.5) + (p_lgb * 0.5)
            
            score = r2_score(np.expm1(y_va), np.expm1(p_ens))
            scores.append(score)
            final_models[target].append({'xgb': m_xgb, 'lgb': m_lgb})
            
        print(f"   평균 R2: {np.mean(scores):.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({'models': final_models, 'features': X.columns.tolist()}, f)
    print(">> 앙상블 모델 저장 완료.")

if __name__ == "__main__":
    train_cv()
