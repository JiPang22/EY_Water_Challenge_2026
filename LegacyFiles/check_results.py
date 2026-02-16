import pickle
import pandas as pd
import numpy as np

MODEL_PATH = "xgb_models_final.pkl"
SUB_PATH = "submission.csv"

def check():
    # 1. 피처 중요도 확인
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        models, features = data['models'], data['features']
    
    print(">> 피처 중요도 분석 (Top 5)")
    for target, model in models.items():
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print(f"\n[{target}]\n{importance.head(5)}")

    # 2. 제출 파일 수치 요약
    if os.path.exists(SUB_PATH):
        df = pd.read_csv(SUB_PATH)
        print("\n>> 제출 파일 상단 5행")
        print(df.head())
        print("\n>> 타겟별 예측값 기초 통계")
        print(df.describe().loc[['min', 'max', 'mean']])

if __name__ == "__main__":
    import os
    check()
