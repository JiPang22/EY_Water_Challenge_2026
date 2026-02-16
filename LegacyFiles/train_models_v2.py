import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
import time
import pickle
import os

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

def train_pipeline(df_final):
    features = [c for c in df_final.columns if c not in TARGET_COLS + ['date']]
    X = df_final[features]
    y = df_final[TARGET_COLS]

    train_idx = int(len(df_final) * 0.8)
    X_train, X_val = X.iloc[:train_idx], X.iloc[train_idx:]
    y_train, y_val = y.iloc[:train_idx], y.iloc[train_idx:]

    models = {}
    scores = {}

    start_time = time.time()
    print(f">> [Fix] XGBoost 2.0 API 대응 학습 시작")

    for i, target in enumerate(TARGET_COLS):
        y_train_log = np.log1p(np.maximum(y_train[target], 0))
        y_val_log = np.log1p(np.maximum(y_val[target], 0))

        # Fix: early_stopping_rounds를 생성자(__init__)로 이동
        model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            tree_method='hist',
            early_stopping_rounds=100  # 여기로 이동
        )

        # eval_set만 fit에 전달
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False
        )

        preds = np.expm1(model.predict(X_val))
        score = r2_score(y_val[target], preds)

        models[target] = model
        scores[target] = score

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(TARGET_COLS) - (i + 1))
        print(f"[{target}] R^2: {score:.4f} | 예상 종료까지: {remaining:.2f}s")

    with open("xgb_models_v2.pkl", "wb") as f:
        pickle.dump(models, f)

    print(f">> 학습 완료. 총 소요: {time.time() - start_time:.2f}s")
    return scores

if __name__ == "__main__":
    from feature_engineering import process_data, generate_mock_data
    df_t, df_l, df_tr = generate_mock_data()
    df_final = process_data(df_t, df_l, df_tr)
    train_pipeline(df_final)
