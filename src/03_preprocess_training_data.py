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

# 스케일링 대상 피처 (Landsat/TerraClimate 7개)
SCALER_FEATURE_COLS = [
    "nir",
    "green",
    "swir16",
    "swir22",
    "NDMI",
    "MNDWI",
    "pet",
]


def build_and_save_tabular_scaler(
    root_dir: str,
    train_csv_name: str = "water_quality_training_dataset.csv",
    landsat_csv_name: str = "landsat_features_training.csv",
    terra_csv_name: str = "terraclimate_features_training.csv",
    scaler_output_name: str = "scaler_tabular.pkl",
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
    landsat_cols = merge_keys + [
        "nir",
        "green",
        "swir16",
        "swir22",
        "NDMI",
        "MNDWI",
    ]

    df = pd.merge(train_df, landsat_df[landsat_cols], on=merge_keys, how="left")
    df = pd.merge(df, terra_df[merge_keys + ["pet"]], on=merge_keys, how="left")

    # 스케일링 대상 칼럼 확인
    missing = [c for c in SCALER_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required scaler features not found in merged training data: {missing}"
        )

    X_train = df[SCALER_FEATURE_COLS].astype(np.float32).values

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
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n💾 Saved tabular scaler to: {scaler_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    build_and_save_tabular_scaler(root_dir)


if __name__ == "__main__":
    main()

