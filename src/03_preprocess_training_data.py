"""
[03단계] 학습용 탭 피처 전처리 & 스케일링

- 입력:
  - water_quality_training_dataset.csv
  - landsat_features_training.csv
  - terraclimate_features_training.csv
- 출력:
  - 전처리된 학습용 CSV (선택 사항)
  - StandardScaler 객체(pkl): scaler_tabular.pkl

이 스크립트는:
1) 학습용 CSV와 외부 피처 CSV를 병합하고
2) 탭 피처(특히 Landsat/TerraClimate 7개 피처)에 대해
   sklearn의 StandardScaler를 사용해 평균 0, 분산 1로 맞춥니다.
3) 추론 단계에서 동일한 스케일링을 재사용할 수 있도록
   학습된 스케일러를 디스크에 저장합니다.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 스케일링 대상 피처 (Landsat/TerraClimate, cloud_cover는 Landsat CSV에 있으면 포함)
SCALER_FEATURE_COLS = [
    "nir", "green", "swir16", "swir22", "NDMI", "MNDWI", "pet"
]

def build_and_save_tabular_scaler(
    root_dir: str,
    train_csv_name: str = "water_quality_training_dataset.csv",
    landsat_csv_name: str = "landsat_features_training.csv",
    terra_csv_name: str = "terraclimate_features_training.csv",
    scaler_output_name: str = "models/scaler_tabular.pkl",
) -> None:
    """
    학습 데이터 기반으로 StandardScaler를 학습하고,
    스케일러 객체를 pickle 파일로 저장합니다.
    """
    raw_dir = os.path.join(root_dir, "rawData")
    train_path = os.path.join(raw_dir, train_csv_name)
    landsat_path = os.path.join(root_dir, landsat_csv_name)
    terra_path = os.path.join(root_dir, terra_csv_name)

    print(f"📂 Train CSV       : {train_path}")
    print(f"📂 Landsat Features: {landsat_path}")
    print(f"📂 Terra Features  : {terra_path}")

    train_df = pd.read_csv(train_path)
    train_df.columns = train_df.columns.str.strip()

    landsat_df = pd.read_csv(landsat_path)
    landsat_df.columns = landsat_df.columns.str.strip()

    terra_df = pd.read_csv(terra_path)
    terra_df.columns = terra_df.columns.str.strip()

    merge_keys = ["Latitude", "Longitude", "Sample Date"]
    # Landsat columns needed: nir, green, swir16, swir22, NDMI, MNDWI
    landsat_cols = merge_keys + ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]

    # 병합 전 디버깅: 날짜 포맷 통일 및 위도/경도 반올림
    print(f"\n🔍 [Debug] Before merge:")
    print(f"  Train rows: {len(train_df)}")
    print(f"  Landsat rows: {len(landsat_df)}")
    print(f"  Terra rows: {len(terra_df)}")
    
    # 위도/경도 반올림 (소수점 6자리로 통일) - 먼저 처리
    for df_temp in [train_df, landsat_df, terra_df]:
        df_temp["Latitude"] = pd.to_numeric(df_temp["Latitude"], errors="coerce").round(6)
        df_temp["Longitude"] = pd.to_numeric(df_temp["Longitude"], errors="coerce").round(6)
    
    # 날짜를 datetime으로 변환 후 다시 문자열로 (포맷 통일)
    # 실패한 날짜는 원본 문자열 유지
    for df_temp in [train_df, landsat_df, terra_df]:
        dates_parsed = pd.to_datetime(df_temp["Sample Date"], dayfirst=True, errors="coerce")
        df_temp["Sample Date"] = dates_parsed.dt.strftime("%d-%m-%Y").fillna(df_temp["Sample Date"])
    
    # 병합 키를 문자열로 변환 (정확한 매칭을 위해)
    for df_temp in [train_df, landsat_df, terra_df]:
        df_temp["Latitude"] = df_temp["Latitude"].astype(str)
        df_temp["Longitude"] = df_temp["Longitude"].astype(str)
        df_temp["Sample Date"] = df_temp["Sample Date"].astype(str)

    # 병합 전 공통 키 확인
    train_keys = set(zip(train_df["Latitude"], train_df["Longitude"], train_df["Sample Date"]))
    landsat_keys = set(zip(landsat_df["Latitude"], landsat_df["Longitude"], landsat_df["Sample Date"]))
    terra_keys = set(zip(terra_df["Latitude"], terra_df["Longitude"], terra_df["Sample Date"]))
    
    common_train_landsat = len(train_keys & landsat_keys)
    common_train_terra = len(train_keys & terra_keys)
    
    print(f"  Common keys (Train & Landsat): {common_train_landsat} / {len(train_keys)}")
    print(f"  Common keys (Train & Terra): {common_train_terra} / {len(train_keys)}")

    df = pd.merge(train_df, landsat_df[landsat_cols], on=merge_keys, how="left")
    matched_landsat = df['nir'].notna().sum()
    print(f"  After Landsat merge: {len(df)} rows, matched: {matched_landsat}")
    
    df = pd.merge(df, terra_df[merge_keys + ["pet"]], on=merge_keys, how="left")
    matched_terra = df['pet'].notna().sum()
    print(f"  After Terra merge: {len(df)} rows, matched: {matched_terra}")
    
    if matched_landsat == 0:
        print("\n⚠️ Landsat 병합 실패! 첫 3행 키 비교:")
        print("Train:", train_df[merge_keys].head(3).values.tolist())
        print("Landsat:", landsat_df[merge_keys].head(3).values.tolist())

    missing = [c for c in SCALER_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required scaler features not found in merged training data: {missing}"
        )

    # NaN이 있는 행은 제외하고 스케일러 학습
    df_valid = df[SCALER_FEATURE_COLS].dropna()
    print(f"\n⚠️ Valid rows for scaler (non-NaN): {len(df_valid)} / {len(df)}")
    
    if len(df_valid) == 0:
        raise ValueError("❌ No valid rows found after merge! Check merge keys.")
    
    X_train = df_valid[SCALER_FEATURE_COLS].astype(np.float32).values

    scaler = StandardScaler()
    scaler.fit(X_train)

    # 간단한 통계 확인 (유닛 테스트 역할)
    X_scaled = scaler.transform(X_train)
    mean_vec = X_scaled.mean(axis=0)
    var_vec = X_scaled.var(axis=0)

    print("\n✅ Scaler validation on training data (mean≈0, var≈1 기대)")
    for name, m, v in zip(SCALER_FEATURE_COLS, mean_vec, var_vec):
        print(f"  - {name}: mean={m:.4f}, var={v:.4f}")

    # 스케일러를 루트 디렉터리에 저장
    scaler_path = os.path.join(root_dir, scaler_output_name)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n💾 Saved tabular scaler to: {scaler_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    build_and_save_tabular_scaler(root_dir)


if __name__ == "__main__":
    main()
