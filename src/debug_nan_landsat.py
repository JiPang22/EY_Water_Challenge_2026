"""
Landsat 피처 NaN 원인 진단 스크립트

어떤 행에서 Landsat 피처가 NaN인지, 그리고 원본 CSV에 데이터가 있는지 확인합니다.
"""
import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)

train_path = os.path.join(root_dir, "rawData", "water_quality_training_dataset.csv")
landsat_path = os.path.join(root_dir, "landsat_features_training.csv")

print("=" * 70)
print("1. 원본 CSV 파일 로드 및 기본 정보")
print("=" * 70)

train_df = pd.read_csv(train_path)
landsat_df = pd.read_csv(landsat_path)

train_df.columns = train_df.columns.str.strip()
landsat_df.columns = landsat_df.columns.str.strip()

print(f"Train CSV: {len(train_df)} rows")
print(f"Landsat CSV: {len(landsat_df)} rows")

# Landsat CSV에서 NaN 행 확인
landsat_nan_count = landsat_df[["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]].isna().all(axis=1).sum()
print(f"\nLandsat CSV에서 모든 피처가 NaN인 행: {landsat_nan_count}")

print("\n" + "=" * 70)
print("2. Landsat CSV에서 NaN인 행 샘플 (처음 10개)")
print("=" * 70)

nan_mask = landsat_df[["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]].isna().all(axis=1)
nan_rows = landsat_df[nan_mask]

if len(nan_rows) > 0:
    print(f"\n총 {len(nan_rows)}개 행이 Landsat CSV에서 NaN입니다.")
    print("\n[NaN 행 샘플 - 처음 10개]")
    print(nan_rows[["Latitude", "Longitude", "Sample Date", "nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]].head(10))
    
    # 해당 행들의 날짜/위치 분포 확인
    print("\n[NaN 행들의 날짜 분포]")
    nan_rows["Sample Date"] = pd.to_datetime(nan_rows["Sample Date"], dayfirst=True, errors="coerce")
    print(nan_rows["Sample Date"].dt.year.value_counts().sort_index())
else:
    print("✅ Landsat CSV에는 NaN 행이 없습니다.")

print("\n" + "=" * 70)
print("3. 병합 후 NaN 발생 확인")
print("=" * 70)

# 병합 키 준비 (03_preprocess와 동일한 방식)
merge_keys = ["Latitude", "Longitude", "Sample Date"]

# 위도/경도 반올림
train_df["Latitude"] = pd.to_numeric(train_df["Latitude"], errors="coerce").round(6)
train_df["Longitude"] = pd.to_numeric(train_df["Longitude"], errors="coerce").round(6)
landsat_df["Latitude"] = pd.to_numeric(landsat_df["Latitude"], errors="coerce").round(6)
landsat_df["Longitude"] = pd.to_numeric(landsat_df["Longitude"], errors="coerce").round(6)

# 날짜 포맷 통일
train_df["Sample Date"] = pd.to_datetime(train_df["Sample Date"], dayfirst=True, errors="coerce").dt.strftime("%d-%m-%Y")
landsat_df["Sample Date"] = pd.to_datetime(landsat_df["Sample Date"], dayfirst=True, errors="coerce").dt.strftime("%d-%m-%Y")

# 문자열로 변환
train_df["Latitude"] = train_df["Latitude"].astype(str)
train_df["Longitude"] = train_df["Longitude"].astype(str)
train_df["Sample Date"] = train_df["Sample Date"].astype(str)
landsat_df["Latitude"] = landsat_df["Latitude"].astype(str)
landsat_df["Longitude"] = landsat_df["Longitude"].astype(str)
landsat_df["Sample Date"] = landsat_df["Sample Date"].astype(str)

# 병합
merged = pd.merge(
    train_df,
    landsat_df[merge_keys + ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]],
    on=merge_keys,
    how="left"
)

merged_nan_count = merged["nir"].isna().sum()
print(f"\n병합 후 nir가 NaN인 행: {merged_nan_count} / {len(merged)}")

if merged_nan_count > 0:
    merged_nan_rows = merged[merged["nir"].isna()]
    print("\n[병합 후 NaN인 행 샘플 - 처음 10개]")
    print(merged_nan_rows[merge_keys + ["nir"]].head(10))
    
    # 원본 Landsat CSV에 해당 키가 있는지 확인
    print("\n[원인 분석: 원본 Landsat CSV에 해당 키가 있는가?]")
    sample_nan_keys = merged_nan_rows[merge_keys].head(5)
    
    for idx, row in sample_nan_keys.iterrows():
        key_tuple = (row["Latitude"], row["Longitude"], row["Sample Date"])
        in_landsat = (landsat_df[merge_keys] == row[merge_keys]).all(axis=1).any()
        print(f"  Key {key_tuple}: Landsat CSV에 존재? {in_landsat}")
        
        if in_landsat:
            # Landsat CSV에서 해당 행 찾기
            match = landsat_df[(landsat_df[merge_keys] == row[merge_keys]).all(axis=1)]
            if len(match) > 0:
                print(f"    → Landsat CSV의 nir 값: {match.iloc[0]['nir']}")

print("\n" + "=" * 70)
print("4. 요약")
print("=" * 70)
print(f"1. Landsat CSV 자체에 NaN인 행: {landsat_nan_count}")
print(f"2. 병합 후 NaN인 행: {merged_nan_count}")
print(f"3. 차이: {merged_nan_count - landsat_nan_count} (병합 과정에서 추가로 발생한 NaN)")

if landsat_nan_count == merged_nan_count:
    print("\n✅ 결론: Landsat CSV 자체에 데이터가 없는 것이 원인입니다.")
    print("   → 위성 데이터 수집 실패 또는 해당 위치/날짜에 데이터가 없음")
elif merged_nan_count > landsat_nan_count:
    print("\n⚠️ 결론: 일부는 Landsat CSV에 없고, 일부는 병합 과정에서 발생했습니다.")
else:
    print("\n❓ 예상치 못한 상황입니다.")
