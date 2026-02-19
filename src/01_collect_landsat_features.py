"""
[01단계] Landsat 위성 피처 수집 스크립트

- 입력: submission_template.csv (또는 테스트용 템플릿 CSV)
- 출력: landsat_features_test.csv
- 역할: 각 관측지(위도/경도/날짜)에 대해 Planetary Computer에서
        Landsat L2 데이터를 조회하고, 밴드/지수를 계산하여 CSV로 저장합니다.

이 파일은 기존 `Landsat_Robust.py`의 최종 버전을
실행 순서가 보이도록 파일명만 재정리한 것입니다.
"""

import pandas as pd
import pystac_client
import planetary_computer
import os
import time
from tqdm import tqdm
import odc.stac
import numpy as np
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# ⚙️ 설정
TEMPLATE_PATH = "submission_template.csv"
OUTPUT_PATH = "landsat_features_test.csv"

# 🚨 실패 시 채워넣을 '빈 사전' 정의 (형태 통일)
NAN_RESULT = {
    'nir': np.nan,
    'green': np.nan,
    'swir16': np.nan,
    'swir22': np.nan,
    'NDMI': np.nan,
    'MNDWI': np.nan,
}


def get_catalog():
    """Planetary Computer STAC API 카탈로그 객체를 생성합니다."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def main():
    """
    1) 템플릿 CSV를 읽고
    2) 각 행의 위도/경도/날짜로 Landsat 장면을 검색한 뒤
    3) nir/green/swir16/swir22, NDMI, MNDWI를 계산하여
       `landsat_features_test.csv`에 저장합니다.
    """
    print("🚀 Starting Robust Landsat Extraction (Step 01)...")

    # 1. 템플릿 CSV 로드 (일반적으로 submission_template.csv)
    if os.path.exists(TEMPLATE_PATH):
        df = pd.read_csv(TEMPLATE_PATH)
    elif os.path.exists(f"../{TEMPLATE_PATH}"):
        df = pd.read_csv(f"../{TEMPLATE_PATH}")
    else:
        print("❌ Template not found")
        return

    results = []
    catalog = get_catalog()

    # 성공/실패 카운트
    success_cnt = 0
    fail_cnt = 0

    # 2. 각 샘플(위도/경도/날짜)에 대해 반복
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            lat = row["Latitude"]
            lon = row["Longitude"]
            date = pd.to_datetime(row["Sample Date"], dayfirst=True)

            # 날짜 범위: 관측일 ± 15일
            start_date = (date - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            end_date = (date + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            time_range = f"{start_date}/{end_date}"

            # 작은 박스(bbox) 정의
            bbox = [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001]

            # 2-1. STAC API로 Landsat 장면 검색
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime=time_range,
                query={"eo:cloud_cover": {"lt": 30}},
            )
            items = search.item_collection()

            if len(items) == 0:
                # 적절한 장면을 찾지 못하면, 미리 정의한 NaN 결과 사용
                results.append(NAN_RESULT)
                fail_cnt += 1
                continue

            # 2-2. 가장 구름이 적은 장면 1개 선택 및 서명
            best_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
            signed_item = planetary_computer.sign(best_item)

            # 2-3. 선택된 장면에서 원하는 밴드를 xarray로 로드
            ds = odc.stac.load(
                [signed_item],
                bands=["nir08", "red", "green", "swir16", "swir22"],
                bbox=bbox,
                resolution=30,
            )

            if ds.sizes["time"] > 0:
                # 공간 차원(x, y)에 대한 중앙값 추출
                data = ds.isel(time=0).median(dim=["x", "y"])

                nir = float(data.nir08)
                green = float(data.green)
                swir16 = float(data.swir16)
                swir22 = float(data.swir22)

                # 물 관련 지수 계산
                mndwi = (green - swir16) / (green + swir16 + 1e-6)
                ndmi = (nir - swir16) / (nir + swir16 + 1e-6)

                results.append(
                    {
                        "nir": nir,
                        "green": green,
                        "swir16": swir16,
                        "swir22": swir22,
                        "NDMI": ndmi,
                        "MNDWI": mndwi,
                    }
                )
                success_cnt += 1
            else:
                # 시간 축에 데이터가 없을 때도 NaN 결과 사용
                results.append(NAN_RESULT)
                fail_cnt += 1

        except Exception as e:
            # 어떤 에러가 나도 NaN 결과를 넣고 계속 진행
            results.append(NAN_RESULT)
            fail_cnt += 1

            # 인증 관련 에러가 나면 카탈로그를 새로 연결
            if "Authentication" in str(e):
                catalog = get_catalog()

    # 3. 결과를 원본 템플릿과 합쳐서 CSV로 저장
    feat_df = pd.DataFrame(results)
    final_df = pd.concat([df, feat_df], axis=1)

    # 결측치 평균값으로 채우기 (모델 입력용)
    final_df = final_df.fillna(final_df.mean(numeric_only=True))

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Final Extraction Done! Saved to {OUTPUT_PATH}")
    print(f"📊 Stats: Success={success_cnt}, Fails={fail_cnt}")


if __name__ == "__main__":
    main()

