"""
[02단계] TerraClimate 기후 피처 수집 스크립트

- 입력: submission_template.csv (또는 테스트용 템플릿 CSV)
- 출력: terraclimate_features_test.csv
- 역할: Planetary Computer의 Terraclimate zarr 데이터셋에서
        PET(potential evapotranspiration) 값을 추출해 CSV로 저장합니다.

이 파일은 기존 `TerraClimate_Robust.py`의 최종 버전을
실행 순서가 보이도록 파일명만 재정리한 것입니다.
"""

import xarray as xr
import pandas as pd
import numpy as np
import pystac_client
import planetary_computer
from tqdm import tqdm
import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# ⚙️ 설정
TEMPLATE_PATH = "submission_template.csv"
OUTPUT_PATH = "terraclimate_features_test.csv"


def get_fresh_ds():
    """
    Planetary Computer에서 Terraclimate 컬렉션을 찾고,
    PET 변수를 포함한 zarr 데이터셋을 xarray로 여는 함수입니다.

    - 이 함수를 매번 호출하는 이유:
      인증 토큰(Access Token)이 만료되거나 연결이 끊겼을 때
      새 연결을 쉽게 만들기 위해서입니다.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-https"]

    # asset.href 안에 이미 인증 토큰이 포함되어 있으며,
    # xarray가 이를 인식해서 zarr 데이터를 불러옵니다.
    ds = xr.open_dataset(asset.href, engine="zarr")
    return ds


def main():
    """
    1) 템플릿 CSV를 읽고
    2) 월 단위로 필요한 PET 값을 모은 뒤
    3) 각 좌표/날짜에 대한 PET를 nearest 방식으로 추출하여
       `terraclimate_features_test.csv`로 저장합니다.
    """
    print(f"📂 Loading {TEMPLATE_PATH}...")
    if os.path.exists(TEMPLATE_PATH):
        df_template = pd.read_csv(TEMPLATE_PATH)
    elif os.path.exists(f"../{TEMPLATE_PATH}"):
        df_template = pd.read_csv(f"../{TEMPLATE_PATH}")
    else:
        print(f"❌ {TEMPLATE_PATH} not found!")
        return

    # 날짜 처리: pandas Period를 사용해 "월" 단위로 그룹화
    df_template["dt"] = pd.to_datetime(df_template["Sample Date"], dayfirst=True)
    unique_dates = df_template["dt"].dt.to_period("M").unique()

    results = []
    print(f"🚀 Starting TerraClimate extraction for {len(unique_dates)} months...")

    error_count = 0
    success_count = 0

    # 1. 각 월(period)에 대해 반복
    for period in tqdm(unique_dates):
        try:
            # 매번 새로운 Dataset을 열어 인증 문제를 회피
            ds = get_fresh_ds()

            # 1-1. 해당 월의 시간 선택 (보통 그 달 1일 기준)
            target_time = period.to_timestamp()
            ds_slice = ds["pet"].sel(time=target_time, method="nearest")

            # 1-2. 현재 월에 해당하는 샘플만 필터링
            mask = df_template["dt"].dt.to_period("M") == period
            current_rows = df_template[mask]
            if len(current_rows) == 0:
                continue

            # 1-3. 좌표 리스트를 xarray DataArray로 만들기
            lats = xr.DataArray(current_rows["Latitude"].values, dims="z")
            lons = xr.DataArray(current_rows["Longitude"].values, dims="z")

            # 1-4. PET 값을 최근접 격자에서 가져오기
            values = ds_slice.sel(lat=lats, lon=lons, method="nearest").values

            # 1-5. 결과 수집
            for idx, val in zip(current_rows.index, values):
                results.append(
                    {
                        "Latitude": df_template.loc[idx, "Latitude"],
                        "Longitude": df_template.loc[idx, "Longitude"],
                        "Sample Date": df_template.loc[idx, "Sample Date"],
                        "pet": float(val),
                    }
                )

            success_count += 1

        except Exception as e:
            error_count += 1
            if error_count == 1:
                print(f"\n⚠️ First Error at {period}: {e}")
            continue

    # 2. 결과를 원본 템플릿과 병합하거나, 완전히 실패했을 경우 대체값 사용
    if len(results) == 0:
        # 어떤 이유로든 전체 추출이 실패했다면, pet을 상수(mean)로 채웁니다.
        print("\n⚠️ ALL FAILED. Using Mean Value (100.0) fallback.")
        final_df = df_template.copy()
        final_df["pet"] = 100.0
    else:
        partial_df = pd.DataFrame(results)
        final_df = pd.merge(
            df_template,
            partial_df,
            on=["Latitude", "Longitude", "Sample Date"],
            how="left",
        )

        # 남은 결측치는 pet 평균으로 채움
        if final_df["pet"].isnull().sum() > 0:
            final_df["pet"] = final_df["pet"].fillna(final_df["pet"].mean())

    if "dt" in final_df.columns:
        final_df = final_df.drop(columns=["dt"])

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Final Extraction Status: Success={success_count}, Errors={error_count}")
    print(f"💾 Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

