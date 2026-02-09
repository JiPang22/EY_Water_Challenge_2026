import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
import time
import pickle

# 타겟 변수 목록
TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

def train_pipeline(df_final):
    features = [c for c in df_final.columns if c not in TARGET_COLS + ['date']]
    X = df_final[features]
    y = df_final[TARGET_COLS]

    # 학습/검증 데이터셋 분리 (8:2)
    train_idx = int(len(df_final) * 0.8)
    X_train, X_val = X.iloc[:train_idx], X.iloc[train_idx:]
    y_train, y_val = y.iloc[:train_idx], y.iloc[train_idx:]

    models = {}
    scores = {}

    start_time = time.time()
    print(f">> 타겟 로그 변환(Log Transform) 적용 학습 시작")

    for i, target in enumerate(TARGET_COLS):
        # 1. 타겟 로그 변환: 수질 데이터 특유의 우측 편향성(Skewness) 및 이상치 영향 최소화
        # $$ y_{transformed} = \ln(y + 1) $$_
        y_train_log = np.log1p(np.maximum(y_train[target], 0))
        y_val_log = np.log1p(np.maximum(y_val[target], 0))

        # 2. 모델 정의 및 인터페이스 설정
        model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            tree_method='hist'
        )

        # 3. 학습 및 Early Stopping 적용
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            early_stopping_rounds=50,
            verbose=False
        )

        # 4. 예측 및 역변환 (Original Scale 복구)
        # $$ \hat{y} = e^{\hat{y}_{transformed}} - 1 $$_
        preds_log = model.predict(X_val)
        preds = np.expm1(preds_log)

        # 5. R2 Score 산출
        score = r2_score(y_val[target], preds)
        models[target] = model
        scores[target] = score

        # 종료 예상 시간 출력
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(TARGET_COLS) - (i + 1))
        print(f"[{target}] R^2: {score:.4f} | 예상 종료까지: {remaining:.2f}s")

    # 모델 저장 (v2)
    with open("xgb_models_v2.pkl", "wb") as f:
        pickle.dump(models, f)

    print(f">> 전체 학습 완료. 총 소요: {time.time() - start_time:.2f}s")
    return scores

if __name__ == "__main__":
    # 단위 테스트 로직
    from feature_engineering import process_data, generate_mock_data
    df_t, df_l, df_tr = generate_mock_data()

    # Phosphorus 타겟에 극단적 스케일링 주입하여 로그 변환 효과 검증
    df_t[TARGET_COLS[2]] = df_t[TARGET_COLS[2]] ** 2

    df_final = process_data(df_t, df_l, df_tr)
    train_pipeline(df_final)
