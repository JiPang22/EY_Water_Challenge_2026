import pandas as pd
import numpy as np
import rasterio
import os
from glob import glob

# 인터페이스 명세
CSV_PATH = "data/processed/train_merged.csv"
TIF_DIR = "data/raw/landsat_images/"  # 실제 경로에 맞춰 수정 필요
SAVE_PATH = "data/processed/train_with_tif.csv"

def get_pixel_value(tif_path, lat, lon):
    try:
        with rasterio.open(tif_path) as src:
            # 좌표계 변환: $$ (lon, lat) \rightarrow (row, col) $$_
            row, col = src.index(lon, lat)
            # 해당 위치의 모든 밴드 값 추출
            return src.read()[:, row, col]
    except Exception:
        return None

def process():
    df = pd.read_csv(CSV_PATH)
    print(f">> 원본 이미지 피처 추출 시작. 대상 행 수: {len(df)}")
    
    # 예시: 가장 가까운 날짜의 TIF 파일을 매칭하는 로직 (단순화)
    # 실제로는 date 컬럼과 tif 파일명을 매칭해야 함
    tif_files = glob(os.path.join(TIF_DIR, "*.tif"))
    if not tif_files:
        print("Error: TIF 파일이 경로에 없음.")
        return

    # 입문자용: 첫 번째 이미지에서 픽셀 값 샘플링 테스트
    sample_tif = tif_files[0]
    new_features = []
    
    for i, row in df.iterrows():
        pixel_vals = get_pixel_value(sample_tif, row['latitude'], row['longitude'])
        if pixel_vals is not None:
            new_features.append(pixel_vals)
        else:
            new_features.append([np.nan] * 7) # Landsat 7개 밴드 기준
            
    # 피처 병합
    tif_df = pd.DataFrame(new_features, columns=[f'band_{i+1}' for i in range(7)])
    df = pd.concat([df, tif_df], axis=1).fillna(df.median(numeric_only=True))
    
    df.to_csv(SAVE_PATH, index=False)
    print(f">> 추출 완료. 저장 경로: {SAVE_PATH}")

if __name__ == "__main__":
    process()
