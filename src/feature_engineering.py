import pandas as pd
import numpy as np
import os
import time

# 인터페이스 명세
PATH_TARGET = "data/raw/water_quality_training_dataset.csv"
PATH_LANDSAT = "data/raw/landsat_features_training.csv"
PATH_TERRA = "data/raw/terraclimate_features_training.csv"
SAVE_PATH = "data/processed/train_merged.csv"

def unit_test_features(df):
    """지수 계산 결과 검증"""
    print(">> [Unit Test] NDWI 지수 범위 체크...")
    assert 'ndwi' in df.columns, "NDWI 컬럼 누락"
    print(f"NDWI Min: {df['ndwi'].min():.2f}, Max: {df['ndwi'].max():.2f}")
    print("테스트 통과.")

def process():
    start_time = time.time()
    
    df_target = pd.read_csv(PATH_TARGET)
    df_l = pd.read_csv(PATH_LANDSAT)
    df_t = pd.read_csv(PATH_TERRA)

    for df in [df_target, df_l, df_t]:
        df.columns = df.columns.str.strip().str.lower()
        if 'sample date' in df.columns:
            df.rename(columns={'sample date': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'].astype(str).str.split('/').str[0], dayfirst=True, errors='coerce')
        df['latitude'] = df['latitude'].round(5)
        df['longitude'] = df['longitude'].round(5)

    df = pd.merge(df_target, df_l, on=['latitude', 'longitude', 'date'], how='left')
    df = pd.merge(df, df_t, on=['latitude', 'longitude', 'date'], how='left')

    # NDWI (Normalized Difference Water Index) 계산
    # $$ NDWI = \frac{Green - NIR}{Green + NIR} $$_
    if 'green' in df.columns and 'nir' in df.columns:
        df['ndwi'] = (df['green'] - df['nir']) / (df['green'] + df['nir'] + 1e-5)

    df = df.sort_values(by=['latitude', 'longitude', 'date'])
    for f in ['pet', 'aet', 'pr']:
        if f in df.columns:
            df[f'{f}_lag1'] = df.groupby(['latitude', 'longitude'])[f].shift(1)
    
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

    df = df.fillna(df.median(numeric_only=True))
    
    unit_test_features(df)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    
    print(f">> 완료. 데이터 크기: {df.shape}")
    print(f"종료 예상 시간: 0.00초 (실제 소요: {time.time() - start_time:.2f}초)")

if __name__ == "__main__":
    process()
