"""
[02.5단계] 수집된 피처 시각화 및 탐색
- 입력:
  - water_quality_training_dataset.csv
  - landsat_features_training.csv
  - terraclimate_features_training.csv
  - rawData/chips/ (이미지 칩)
- 출력:
  - plots/02.5_image_chip_samples.png (10개 위성 이미지 칩 샘플)
  - plots/02.5_temporal_distribution_2014-2015.png (2014-2015년 월별 데이터 분포)
  - plots/02.5_temporal_distribution_2014_Jan-Feb.png (2014년 1~2월 일별 데이터 분포)
- 역할:
  데이터 수집(01, 02)과 전처리(03) 사이에, 수집된 피처가
  제대로 병합되었는지, 어떤 분포를 가지는지 시각적으로 확인합니다.
"""

import os

import matplotlib

# GUI 없는 서버 환경에서도 그래프를 이미지 파일로 저장하기 위한 설정
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_and_merge_data(root_dir: str) -> pd.DataFrame:
    """
    학습용 원본 CSV와 수집된 Landsat, TerraClimate 피처 CSV를 병합합니다.
    03_preprocess_training_data.py의 병합 로직을 재사용하여 일관성을 유지합니다.
    """
    train_path = os.path.join(root_dir, "rawData", "water_quality_training_dataset.csv")
    landsat_path = os.path.join(root_dir, "landsat_features_training.csv")
    terra_path = os.path.join(root_dir, "terraclimate_features_training.csv")

    # 1. 데이터 로드
    train_df = pd.read_csv(train_path)
    landsat_df = pd.read_csv(landsat_path)
    terra_df = pd.read_csv(terra_path)

    # 2. 병합 키 포맷 통일 (가장 중요)
    for df in [train_df, landsat_df, terra_df]:
        df.columns = df.columns.str.strip()
        # 위도/경도: 소수점 6자리로 반올림하여 부동소수점 오류 방지
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").round(6)
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").round(6)
        # 날짜: '일-월-년' 형식의 문자열로 통일
        df["Sample Date"] = pd.to_datetime(
            df["Sample Date"], dayfirst=True, errors="coerce"
        ).dt.strftime("%d-%m-%Y")

    # 3. 데이터 병합 (left join)
    merge_keys = ["Latitude", "Longitude", "Sample Date"]
    landsat_cols = merge_keys + [
        "nir", "green", "swir16", "swir22", "NDMI", "MNDWI"
    ]
    
    # cloud_cover는 있을 수도 있고 없을 수도 있으므로, 있는 컬럼만 선택
    if 'cloud_cover' in landsat_df.columns:
        landsat_cols.append('cloud_cover')

    df = pd.merge(train_df, landsat_df[landsat_cols], on=merge_keys, how="left")
    df = pd.merge(df, terra_df[merge_keys + ["pet"]], on=merge_keys, how="left")

    print(f"✅ 데이터 병합 완료. 최종 행 수: {len(df)}")
    print(f"  - Landsat 피처 매칭 수: {df['nir'].notna().sum()}")
    print(f"  - TerraClimate 피처 매칭 수: {df['pet'].notna().sum()}")

    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    plot_dir = os.path.join(root_dir, "plots")
    chips_dir = os.path.join(root_dir, "rawData", "chips")
    os.makedirs(plot_dir, exist_ok=True)

    df = load_and_merge_data(root_dir)
    df["Sample Date DT"] = pd.to_datetime(df["Sample Date"], dayfirst=True, errors="coerce")

    # --- 1. 10개 무작위 위성 이미지 칩 시각화 (2x5 그리드) ---
    n_samples = 10
    n_rows, n_cols = 2, 5
    sample_df = df.dropna(subset=["Sample Date DT"]).sample(n=n_samples, random_state=42)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axes = axes.flatten()

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        ax = axes[i]
        try:
            lat_f = float(row["Latitude"])
            lon_f = float(row["Longitude"])
            loc_id = f"{lat_f:.4f}_{lon_f:.4f}"
            date_str = row["Sample Date DT"].strftime("%Y-%m-%d")
            
            chip_path = os.path.join(chips_dir, f"loc_{loc_id}", f"{date_str}.npy")
            
            if os.path.exists(chip_path):
                # 이미지 로드 (5, 32, 32) -> RGB (3, 32, 32) 선택
                rgb_img = np.load(chip_path).astype(np.float32)[:3, :, :]
                
                # 시각화를 위한 정규화 (채널별 2-98 percentile 클리핑)
                rgb_img_viz = np.zeros_like(rgb_img)
                for channel in range(3):
                    c_min, c_max = np.percentile(rgb_img[channel, :, :], [2, 98])
                    rgb_img_viz[channel, :, :] = np.clip(rgb_img[channel, :, :], c_min, c_max)
                    if (c_max - c_min) > 1e-6:
                        rgb_img_viz[channel, :, :] = (rgb_img_viz[channel, :, :] - c_min) / (c_max - c_min)

                # (C, H, W) -> (H, W, C) for imshow
                rgb_img_viz = np.transpose(rgb_img_viz, (1, 2, 0))
                
                ax.imshow(rgb_img_viz)
                ax.set_title(date_str, fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Chip', ha='center', va='center', color='white', backgroundcolor='black')
                ax.set_title(row["Sample Date DT"].strftime("%Y-%m-%d"), fontsize=8, color='r')

        except Exception as e:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            print(f"Error processing row {idx}: {e}")
        ax.axis('off')

    fig.suptitle("Random Sample of 10 Satellite Image Chips (RGB)", fontsize=16)
    path = os.path.join(plot_dir, "02.5_image_chip_samples.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path)
    plt.close()
    print(f"🖼️ 10개 위성 이미지 칩 시각화 저장 완료: {path}")

    # --- 2. 2014-2015년 월별 데이터 분포 시각화 ---
    df_filtered = df[(df["Sample Date DT"].dt.year >= 2014) & (df["Sample Date DT"].dt.year <= 2015)].copy()
    
    if not df_filtered.empty:
        df_filtered['YearMonth'] = df_filtered['Sample Date DT'].dt.strftime('%Y-%m')
        
        plt.figure(figsize=(15, 7))
        sns.countplot(data=df_filtered.sort_values('YearMonth'), x='YearMonth', palette='viridis')
        plt.title("Temporal Distribution of Samples (2014-2015)")
        plt.xlabel("Year-Month")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        path = os.path.join(plot_dir, "02.5_temporal_distribution_2014-2015.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"📊 2014-2015년 월별 분포 시각화 저장 완료: {path}")
    else:
        print("⚠️ 2014-2015년 데이터가 없어 시간 분포 그래프를 생성하지 않았습니다.")

    # --- 3. 2014년 1월 ~ 2월 데이터 분포 시각화 ---
    df_jan_feb = df[
        (df["Sample Date DT"] >= "2014-01-01") & 
        (df["Sample Date DT"] <= "2014-02-28")
    ].copy()
    
    if not df_jan_feb.empty:
        plt.figure(figsize=(12, 6))
        df_jan_feb['DateStr'] = df_jan_feb['Sample Date DT'].dt.strftime('%Y-%m-%d')
        sns.countplot(data=df_jan_feb.sort_values('DateStr'), x='DateStr', palette='viridis')
        plt.title("Daily Distribution of Samples (2014-01 ~ 2014-02)")
        plt.xlabel("Date")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=90)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        path = os.path.join(plot_dir, "02.5_temporal_distribution_2014_Jan-Feb.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"📊 2014년 1~2월 일별 분포 시각화 저장 완료: {path}")
    else:
        print("⚠️ 2014년 1~2월 데이터가 없어 그래프를 생성하지 않았습니다.")


if __name__ == "__main__":
    main()