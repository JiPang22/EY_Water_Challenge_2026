"""
[05단계] 학습 데이터 EDA & 품질 점검

- 입력:
  - water_quality_training_dataset.csv
  - landsat_features_training.csv
  - terraclimate_features_training.csv
- 출력:
  - 간단한 통계/히스토그램/상관계수 플롯 (plots/ 아래 저장)

목적:
- 학습 전에 데이터의 분포와 이상치를 눈으로 확인하기 위한 단계입니다.
- 이 스크립트는 모델을 학습하지 않고, 데이터 품질을 점검하는 데만 집중합니다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_merged_training_dataframe(root_dir: str) -> pd.DataFrame:
    """학습용 CSV와 외부 피처를 병합한 단일 DataFrame을 반환합니다."""
    raw_dir = os.path.join(root_dir, "rawData")
    train_path = os.path.join(raw_dir, "water_quality_training_dataset.csv")
    landsat_path = os.path.join(root_dir, "landsat_features_training.csv")
    terra_path = os.path.join(root_dir, "terraclimate_features_training.csv")

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

    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    plot_dir = os.path.join(root_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    df = load_merged_training_dataframe(root_dir)

    feature_cols = [
        "nir",
        "green",
        "swir16",
        "swir22",
        "NDMI",
        "MNDWI",
        "pet",
    ]
    target_cols = [
        "Total Alkalinity",
        "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]

    print("📊 Basic statistics for features and targets:")
    print(df[feature_cols + target_cols].describe())

    # 히스토그램: 각 피처/타겟 분포
    for col in feature_cols + target_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, bins=40)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        path = os.path.join(plot_dir, f"hist_{col.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # 상관계수 히트맵 (피처+타겟)
    corr_cols = feature_cols + target_cols
    corr = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix (Features + Targets)")
    path = os.path.join(plot_dir, "corr_features_targets.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"📁 EDA plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()

