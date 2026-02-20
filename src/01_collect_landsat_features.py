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

# ⚙️ 설정 (명령행 인자로 덮어쓰기 가능)
TEMPLATE_PATH = "submission_template.csv"
OUTPUT_PATH = "landsat_features_test.csv"

# 학습용: rawData/water_quality_training_dataset.csv → landsat_features_training.csv

# 데이터가 전혀 없을 때만 사용 (형태 통일)
# cloud_cover 포함: 구름 정보도 데이터로 취급
NAN_RESULT = {
    "nir": np.nan,
    "green": np.nan,
    "swir16": np.nan,
    "swir22": np.nan,
    "NDMI": np.nan,
    "MNDWI": np.nan,
    "cloud_cover": np.nan,
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

    # 1. 템플릿 CSV 로드
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        TEMPLATE_PATH,
        os.path.join(base, TEMPLATE_PATH),
        os.path.join(os.getcwd(), TEMPLATE_PATH),
    ]
    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break
    if not found:
        print(f"❌ Template not found. Tried: {candidates}")
        return
    df = pd.read_csv(found)

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

            # 작은 박스(bbox) 정의
            bbox = [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001]

            # 2-1. STAC API로 Landsat 장면 검색 (구름 제한 없음 - 모든 장면 수집)
            # 구름이 많은 장면도 데이터: cloud_cover를 피처로 저장
            items = []
            for days in [15, 30, 60, 90]:
                start_date = (date - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
                end_date = (date + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
                time_range = f"{start_date}/{end_date}"
                search = catalog.search(
                    collections=["landsat-c2-l2"],
                    bbox=bbox,
                    datetime=time_range,
                )
                items = list(search.item_collection())
                if len(items) > 0:
                    break

            if len(items) == 0:
                # 날짜 범위를 넓혀도 장면이 없을 때만 NaN (버리지 않고 기록)
                results.append(NAN_RESULT)
                fail_cnt += 1
                continue

            # 2-2. 가장 구름이 적은 장면 1개 선택 (구름 많은 것도 사용 가능)
            best_item = min(items, key=lambda item: item.properties.get("eo:cloud_cover", 100))
            signed_item = planetary_computer.sign(best_item)
            cloud_cover = float(best_item.properties.get("eo:cloud_cover", np.nan))

            # 2-3. 선택된 장면에서 원하는 밴드를 xarray로 로드
            try:
                ds = odc.stac.load(
                    [signed_item],
                    bands=["nir08", "red", "green", "swir16", "swir22"],
                    bbox=bbox,
                    resolution=30,
                )
            except Exception:
                ds = None

            if ds is not None and ds.sizes["time"] > 0:
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
                        "cloud_cover": cloud_cover,
                    }
                )
                success_cnt += 1
            else:
                # 밴드 로드 실패해도 cloud_cover는 저장 (버리지 않음)
                results.append(
                    {
                        "nir": np.nan,
                        "green": np.nan,
                        "swir16": np.nan,
                        "swir22": np.nan,
                        "NDMI": np.nan,
                        "MNDWI": np.nan,
                        "cloud_cover": cloud_cover,
                    }
                )
                success_cnt += 1

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

    out_path = OUTPUT_PATH if os.path.isabs(OUTPUT_PATH) else os.path.join(base, OUTPUT_PATH)
    final_df.to_csv(out_path, index=False)
    print(f"\n✅ Final Extraction Done! Saved to {out_path}")
    print(f"📊 Stats: Success={success_cnt}, Fails={fail_cnt}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Landsat 피처 수집 (구름 제한 없음)")
    parser.add_argument("--train", action="store_true", help="학습용: water_quality_training_dataset.csv → landsat_features_training.csv")
    parser.add_argument("--template", type=str, help="입력 CSV 경로")
    parser.add_argument("--output", type=str, help="출력 CSV 경로")
    args = parser.parse_args()

    if args.train:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TEMPLATE_PATH = os.path.join(base, "rawData", "water_quality_training_dataset.csv")
        OUTPUT_PATH = os.path.join(base, "landsat_features_training.csv")
        print(f"📂 [학습용] 입력: {TEMPLATE_PATH}\n   출력: {OUTPUT_PATH}")
    if args.template:
        TEMPLATE_PATH = args.template
    if args.output:
        OUTPUT_PATH = args.output

    main()

