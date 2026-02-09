import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import time

# 인터페이스 명세
INPUT_PATH = "data/processed/train_merged.csv"
MODEL_PATH = "water_model.pkl"
TARGET_COLS = ['total alkalinity', 'electrical conductance', 'dissolved reactive phosphorus']
# 학습에 방해되는 정보(날짜, 좌표, ID)는 제외
DROP_COLS = ['date', 'latitude', 'longitude', 'id']

def train():
    df = pd.read_csv(INPUT_PATH)
    
    # 1. 데이터 분리
    # 타겟값과 그와 관련된 로그값들을 모두 제거하여 순수 피처(X)만 남김
    leakage_cols = [c for c in df.columns if c.endswith('_log')]
    X = df.drop(columns=TARGET_COLS + DROP_COLS + leakage_cols, errors='ignore')
    y = df[TARGET_COLS]
    
    # 공부용 80%, 시험용 20% 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    print(f">> 학습 시작. 피처 수: {X.shape[1]}")
    start_time = time.time()

    for i, target in enumerate(TARGET_COLS):
        # 수치 안정성을 위해 로그 변환 적용
        # $$ y_{log} = \ln(y + 1) $$_
        y_tr_log = np.log1p(np.maximum(y_train[target], 0))
        y_va_log = np.log1p(np.maximum(y_val[target], 0))
        
        # 모델 설정 (최적화된 파라미터 반영)
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1, # R5 5600 모든 코어 사용
            random_state=42,
            early_stopping_rounds=50
        )
        
        # 학습
        model.fit(X_train, y_tr_log, eval_set=[(X_val, y_va_log)], verbose=False)
        
        # 평가 (로그를 다시 원래 숫자로 복구)
        preds = np.expm1(model.predict(X_val))
        r2 = r2_score(y_val[target], preds)
        
        models[target] = model
        print(f"[{target}] R2 Score: {r2:.4f}")

    # 결과 저장
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({'models': models, 'features': X.columns.tolist()}, f)
    
    print(f">> 완료. 소요 시간: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train()
