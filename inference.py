import pandas as pd
import numpy as np
import pickle
import os

# 인터페이스 명세
VAL_LANDSAT = "data/raw/landsat_features_validation.csv"
VAL_TERRA = "data/raw/terraclimate_features_validation.csv"
TEMPLATE_PATH = "data/raw/submission_template.csv"
MODEL_PATH = "water_model.pkl"
SAVE_PATH = "submission.csv"

def run_inference():
    # 1. 모델 및 피처 리스트 로드
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        models = data['models']
        expected_features = data['features']

    # 2. 데이터 로드 및 정규화
    df_l = pd.read_csv(VAL_LANDSAT)
    df_t = pd.read_csv(VAL_TERRA)
    df_template = pd.read_csv(TEMPLATE_PATH)

    for df in [df_l, df_t, df_template]:
        df.columns = df.columns.str.strip().str.lower()
        if 'sample date' in df.columns: df.rename(columns={'sample date': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'].astype(str).str.split('/').str[0], dayfirst=True, errors='coerce')
        df['latitude'], df['longitude'] = df['latitude'].round(5), df['longitude'].round(5)

    # 3. 데이터 병합
    df_val = pd.merge(df_template, df_l, on=['latitude', 'longitude', 'date'], how='left')
    df_val = pd.merge(df_val, df_t, on=['latitude', 'longitude', 'date'], how='left')

    # 4. 피처 엔지니어링 (Train과 동일하게 수행)
    if 'green' in df_val.columns and 'nir' in df_val.columns:
        df_val['ndwi'] = (df_val['green'] - df_val['nir']) / (df_val['green'] + df_val['nir'] + 1e-5)
    
    df_val['month_sin'] = np.sin(2 * np.pi * df_val['date'].dt.month / 12)
    df_val['month_cos'] = np.cos(2 * np.pi * df_val['date'].dt.month / 12)
    
    # 시차 변수 (Lag)
    df_val = df_val.sort_values(by=['latitude', 'longitude', 'date'])
    for f in ['pet', 'aet', 'pr']:
        if f in df_val.columns:
            df_val[f'{f}_lag1'] = df_val.groupby(['latitude', 'longitude'])[f].shift(1)

    # 5. 피처 정렬 및 결측치 처리
    # 학습 때 없던 컬럼은 0으로 채우고, 순서를 동일하게 맞춤
    for col in expected_features:
        if col not in df_val.columns: df_val[col] = 0
    
    X_val = df_val[expected_features].fillna(0)

    # 6. 예측 및 로그 역변환
    # $$ \hat{y} = \exp(\text{pred}_{log}) - 1 $$_
    for target, model in models.items():
        preds_log = model.predict(X_val)
        preds = np.expm1(preds_log)
        # 템플릿의 컬럼명 대소문자 무관하게 매칭하여 삽입
        col_name = [c for c in df_template.columns if c.lower() == target][0]
        df_template[col_name] = preds

    # 7. 결과 저장
    df_template.to_csv(SAVE_PATH, index=False)
    print(f">> 제출 파일 생성 완료: {SAVE_PATH}")

if __name__ == "__main__":
    run_inference()
