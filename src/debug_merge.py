"""
병합 실패 원인 진단 스크립트
"""
import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)

train_path = os.path.join(root_dir, "rawData", "water_quality_training_dataset.csv")
landsat_path = os.path.join(root_dir, "landsat_features_training.csv")
terra_path = os.path.join(root_dir, "terraclimate_features_training.csv")

print("=" * 60)
print("1. 파일 존재 여부 확인")
print("=" * 60)
print(f"Train CSV: {os.path.exists(train_path)}")
print(f"Landsat CSV: {os.path.exists(landsat_path)}")
print(f"Terra CSV: {os.path.exists(terra_path)}")

print("\n" + "=" * 60)
print("2. 각 CSV의 행 수 및 컬럼 확인")
print("=" * 60)

train_df = pd.read_csv(train_path)
landsat_df = pd.read_csv(landsat_path)
terra_df = pd.read_csv(terra_path)

print(f"Train: {len(train_df)} rows, columns: {train_df.columns.tolist()}")
print(f"Landsat: {len(landsat_df)} rows, columns: {landsat_df.columns.tolist()}")
print(f"Terra: {len(terra_df)} rows, columns: {terra_df.columns.tolist()}")

print("\n" + "=" * 60)
print("3. 병합 키(첫 5행) 비교")
print("=" * 60)

merge_keys = ["Latitude", "Longitude", "Sample Date"]

print("\n[Train CSV - 첫 5행]")
print(train_df[merge_keys].head())

print("\n[Landsat CSV - 첫 5행]")
print(landsat_df[merge_keys].head())

print("\n[Terra CSV - 첫 5행]")
print(terra_df[merge_keys].head())

print("\n" + "=" * 60)
print("4. 데이터 타입 확인")
print("=" * 60)
print("\n[Train]")
print(train_df[merge_keys].dtypes)
print("\n[Landsat]")
print(landsat_df[merge_keys].dtypes)
print("\n[Terra]")
print(terra_df[merge_keys].dtypes)

print("\n" + "=" * 60)
print("5. 위도/경도 값 정밀도 비교 (첫 행)")
print("=" * 60)
print(f"Train 첫 행: Lat={train_df.iloc[0]['Latitude']}, Lon={train_df.iloc[0]['Longitude']}")
print(f"Landsat 첫 행: Lat={landsat_df.iloc[0]['Latitude']}, Lon={landsat_df.iloc[0]['Longitude']}")
print(f"Terra 첫 행: Lat={terra_df.iloc[0]['Latitude']}, Lon={terra_df.iloc[0]['Longitude']}")

# 정확히 같은지 확인
print("\n첫 행 키가 정확히 일치하는가?")
train_first = tuple(train_df[merge_keys].iloc[0])
landsat_first = tuple(landsat_df[merge_keys].iloc[0])
terra_first = tuple(terra_df[merge_keys].iloc[0])

print(f"Train == Landsat: {train_first == landsat_first}")
print(f"Train == Terra: {train_first == terra_first}")

print("\n" + "=" * 60)
print("6. 실제 병합 시도 (원본 그대로)")
print("=" * 60)

df1 = pd.merge(train_df, landsat_df[merge_keys + ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]], 
               on=merge_keys, how="left")
print(f"Train + Landsat 병합 후: {len(df1)} rows")
print(f"  - nir가 NaN인 행: {df1['nir'].isna().sum()} / {len(df1)}")
print(f"  - nir가 채워진 행: {df1['nir'].notna().sum()} / {len(df1)}")

if df1['nir'].notna().sum() == 0:
    print("\n⚠️ 병합 실패! 첫 3행의 키 비교:")
    print("\n[Train]")
    print(train_df[merge_keys].head(3))
    print("\n[Landsat]")
    print(landsat_df[merge_keys].head(3))
    
    # 공통 키 찾기
    train_keys_set = set(zip(train_df['Latitude'], train_df['Longitude'], train_df['Sample Date']))
    landsat_keys_set = set(zip(landsat_df['Latitude'], landsat_df['Longitude'], landsat_df['Sample Date']))
    common_keys = train_keys_set & landsat_keys_set
    print(f"\n공통 키 개수: {len(common_keys)} / Train: {len(train_keys_set)}, Landsat: {len(landsat_keys_set)}")
    
    if len(common_keys) == 0:
        print("\n❌ 공통 키가 하나도 없습니다!")
        print("첫 3개 키 샘플:")
        print("Train:", list(train_keys_set)[:3])
        print("Landsat:", list(landsat_keys_set)[:3])

print("\n" + "=" * 60)
print("7. 날짜 포맷 변환 후 병합 시도")
print("=" * 60)

train_df2 = train_df.copy()
landsat_df2 = landsat_df.copy()
terra_df2 = terra_df.copy()

# 날짜를 datetime으로 변환 후 다시 문자열로
for df_temp in [train_df2, landsat_df2, terra_df2]:
    df_temp["Sample Date"] = pd.to_datetime(
        df_temp["Sample Date"], dayfirst=True, errors="coerce"
    ).dt.strftime("%d-%m-%Y")

df2 = pd.merge(train_df2, landsat_df2[merge_keys + ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]], 
               on=merge_keys, how="left")
print(f"날짜 포맷 통일 후 병합: {len(df2)} rows")
print(f"  - nir가 채워진 행: {df2['nir'].notna().sum()} / {len(df2)}")

if df2['nir'].notna().sum() > 0:
    print("✅ 날짜 포맷 통일로 해결됨!")
else:
    print("❌ 여전히 실패. 위도/경도 문제일 수 있음.")
