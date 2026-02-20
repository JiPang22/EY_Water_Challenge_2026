# Generated and Patched Extractor for Inference

import warnings
warnings.filterwarnings("ignore")

# 데이터 조작 및 분석
import numpy as np
import pandas as pd

# STAC API 액세스 및 인증을 위한 Planetary Computer 도구
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from pystac.extensions.eo import EOExtension as eo

from datetime import date
from tqdm import tqdm
import os

# 설정
tqdm.pandas()

def compute_Landsat_values(row):
    lat = row['Latitude']
    lon = row['Longitude']
    date = pd.to_datetime(row['Sample Date'], dayfirst=True, errors='coerce')

    # ~100m 버퍼 크기
    bbox_size = 0.00089831  
    bbox = [
        lon - bbox_size / 2,
        lat - bbox_size / 2,
        lon + bbox_size / 2,
        lat + bbox_size / 2
    ]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    # 더 넓은 검색 범위, 나중에 가장 가까운 날짜로 필터링
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime="2011-01-01/2015-12-31",
        query={"eo:cloud_cover": {"lt": 10}},
    )
    
    items = search.item_collection()

    if not items:
        return pd.Series({
            "nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan
        })

    try:
        # 샘플 날짜를 UTC로 변환
        sample_date_utc = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")

        # 샘플 날짜에 가장 가까운 항목 선택
        items = sorted(
            items,
            key=lambda x: abs(pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc)
        )
        selected_item = pc.sign(items[0])

        # 필요한 밴드 로드
        bands_of_interest = ["green", "nir08", "swir16", "swir22"]
        data = stac_load([selected_item], bands=bands_of_interest, bbox=bbox).isel(time=0)

        green = data["green"].astype("float")
        nir = data["nir08"].astype("float")
        swir16 = data["swir16"].astype("float")
        swir22 = data["swir22"].astype("float")
        
        # 중앙값 계산
        median_green = float(green.median(skipna=True).values)
        median_nir = float(nir.median(skipna=True).values)
        median_swir16 = float(swir16.median(skipna=True).values)
        median_swir22 = float(swir22.median(skipna=True).values)

        # 0을 NaN으로 바꾸기
        median_green = median_green if median_green != 0 else np.nan
        median_nir = median_nir if median_nir != 0 else np.nan
        median_swir16 = median_swir16 if median_swir16 != 0 else np.nan
        median_swir22 = median_swir22 if median_swir22 != 0 else np.nan
        
        return pd.Series({
            "nir": median_nir,
            "green": median_green,
            "swir16": median_swir16,
            "swir22": median_swir22,
        })
    
    except Exception as e:
        return pd.Series({
            "nir": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan
        })


Water_Quality_df=pd.read_csv('water_quality_training_dataset.csv')
Water_Quality_df.head()

Water_Quality_df.shape

Water_Quality_df_200 = Water_Quality_df.loc[0:199]
Water_Quality_df_200.shape

# 훈련 데이터셋에 대한 Landsat에서 밴드 값 추출
train_features_path = "landsat_features_training_200.csv"

print("🚀 훈련 데이터에 대한 Landsat 특징 추출 실행 중...")
landsat_train_features = Water_Quality_df_200.progress_apply(compute_Landsat_values, axis=1)
landsat_train_features.to_csv(train_features_path, index=False)

# 지수 생성: NDMI 및 MNDWI
eps = 1e-10
landsat_train_features['NDMI'] = (landsat_train_features['nir'] - landsat_train_features['swir16']) / (landsat_train_features['nir'] + landsat_train_features['swir16'] + eps)
landsat_train_features['MNDWI'] = (landsat_train_features['green'] - landsat_train_features['swir16']) / (landsat_train_features['green'] + landsat_train_features['swir16'] + eps)

landsat_train_features['Latitude'] = Water_Quality_df['Latitude']
landsat_train_features['Longitude'] = Water_Quality_df['Longitude']
landsat_train_features['Sample Date'] = Water_Quality_df['Sample Date']
landsat_train_features = landsat_train_features[['Latitude', 'Longitude', 'Sample Date', 'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']]

landsat_train_features.to_csv(train_features_path, index=False)

# 파일 미리보기
landsat_train_features.head()

Validation_df=pd.read_csv('submission_template.csv')
Validation_df.head()

Validation_df.shape

# 제출 데이터셋에 대한 Landsat에서 밴드 값 추출
val_features_path = "landsat_features_validation.csv"

print("🚀 검증 데이터에 대한 Landsat 특징 추출 실행 중...")
landsat_val_features = Validation_df.progress_apply(compute_Landsat_values, axis=1)
print('Saving to ../landsat_features_test.csv...'); landsat_val_features.to_csv(val_features_path, index=False)

# 지수 생성: NDMI 및 MNDWI
eps = 1e-10
landsat_val_features['NDMI'] = (landsat_val_features['nir'] - landsat_val_features['swir16']) / (landsat_val_features['nir'] + landsat_val_features['swir16'])
landsat_val_features['MNDWI'] = (landsat_val_features['green'] - landsat_val_features['swir16']) / (landsat_val_features['green'] + landsat_val_features['swir16'] + eps)

landsat_val_features['Latitude'] = Validation_df['Latitude']
landsat_val_features['Longitude'] = Validation_df['Longitude']
landsat_val_features['Sample Date'] = Validation_df['Sample Date']
landsat_val_features = landsat_val_features[['Latitude', 'Longitude', 'Sample Date', 'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']]

print('Saving to ../landsat_features_test.csv...'); landsat_val_features.to_csv(val_features_path, index=False)

# 파일 미리보기
landsat_val_features.head()

