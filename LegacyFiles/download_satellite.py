import pandas as pd
import numpy as np
import pystac_client
import planetary_computer
import requests
import os
import time
from tqdm import tqdm

# 인터페이스: CSV(Lat, Lon, Date) -> Microsoft PC STAC -> GeoTIFF
CSV_PATH = "data/raw/train.csv"
SAVE_DIR = "/mnt/data_lake/EY_Satellite_Raw"
BANDS = ["B02", "B03", "B04", "B08"]

def test_stac_connection():
    print("[Test] Microsoft Planetary Computer STAC API 연결 테스트...")
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=[18.0, -35.0, 33.0, -22.0],
            max_items=1
        )
        items = list(search.items()) # Fix: items() 사용
        assert len(items) > 0
        print("[Test] 연결 및 검색 성공.")
        return catalog
    except Exception as e:
        print(f"[Test] 실패: {e}")
        return None

def download_file(url, path):
    if os.path.exists(path):
        return
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
    except:
        pass

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    catalog = test_stac_connection()
    if not catalog: return

    df = pd.read_csv(CSV_PATH)
    
    # 컬럼명 확인 후 매핑 (latitude_, longitude_, date_)
    df['date'] = pd.to_datetime(df['date'])
    unique_points = df[['latitude', 'longitude', 'date']].drop_duplicates()
    total_len = len(unique_points)

    print(f">> 총 {total_len}개 지점 데이터 적재 시작 (Path: {SAVE_DIR})")
    start_time = time.time()

    for idx, row in unique_points.iterrows():
        lat, lon, dt = row['latitude'], row['longitude'], row['date']
        bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
        time_query = f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_query,
            query={"eo:cloud_cover": {"lt": 25}}
        )
        
        items = list(search.items()) # Fix: items() 사용
        if not items: continue

        item = min(items, key=lambda x: x.properties["eo:cloud_cover"])
        signed_item = planetary_computer.sign(item)

        for b in BANDS:
            asset = signed_item.assets.get(b)
            if not asset: continue
            
            f_name = f"SA_{lat:.5f}_{lon:.5f}_{dt.strftime('%Y%m%d')}_{b}.tif"
            f_path = os.path.join(SAVE_DIR, f_name)
            download_file(asset.href, f_path)

        elapsed = time.time() - start_time
        avg_per_point = elapsed / (idx + 1)
        etc_seconds = avg_per_point * (total_len - (idx + 1))
        
        if idx % 5 == 0:
            print(f"[{idx+1}/{total_len}] 처리 중 | 소요: {elapsed/60:.1f}분 | 예상 종료: {etc_seconds/60:.1f}분 후")

    print(f">> 다운로드 종료. 소요 시간: {(time.time()-start_time)/60:.2f}분")

if __name__ == "__main__":
    main()
