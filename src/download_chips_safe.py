import pandas as pd
import pystac_client
import planetary_computer as pc
import rioxarray
import os
import numpy as np
import time
import random
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TRAIN_CSV = "data/raw/water_quality_training_dataset.csv"
SAVE_DIR = "data/raw/satellite_chips/"
CHIP_SIZE = 32
MAX_WORKERS = 4 

def check_integrity(file_path):
    """무결성 검사: 파일 존재, 크기 > 0, 읽기 가능 여부 확인"""
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    try:
        # 파일 헤더만 살짝 읽어서 깨졌는지 확인
        with rioxarray.open_rasterio(file_path) as src:
            pass
        return True
    except:
        return False

def download_one_point(row):
    assets = ["red", "green", "blue", "nir08", "swir16"]
    row_id = row.get('id', row.name)
    
    # 무결성 검사: 5개 밴드 모두 정상인지 확인
    if all(check_integrity(f"{SAVE_DIR}id_{row_id}_{a}.tif") for a in assets):
        return "skipped"

    time.sleep(random.uniform(0.5, 1.5))

    try:
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        lat, lon = row['latitude'], row['longitude']
        date = pd.to_datetime(row['date'])
        date_range = f"{date.year}-01-01/{date.year}-12-31"

        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=[lon-0.005, lat-0.005, lon+0.005, lat+0.005],
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 30}}
        )
        items = search.item_collection()
        if not items: return "no_items"
        
        item = pc.sign(sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[0])
        
        for asset in assets:
            save_path = f"{SAVE_DIR}id_{row_id}_{asset}.tif"
            # 개별 밴드 무결성 검사 후 불합격 시만 다운로드
            if check_integrity(save_path): continue
            
            ds = rioxarray.open_rasterio(item.assets[asset].href)
            transformer = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
            x_utm, y_utm = transformer.transform(lon, lat)
            
            center_ds = ds.sel(x=x_utm, y=y_utm, method="nearest")
            x_idx = np.where(ds.x == center_ds.x)[0][0]
            y_idx = np.where(ds.y == center_ds.y)[0][0]
            
            crop = ds.isel(
                x=slice(max(0, x_idx - CHIP_SIZE // 2), x_idx + CHIP_SIZE // 2),
                y=slice(max(0, y_idx - CHIP_SIZE // 2), y_idx + CHIP_SIZE // 2)
            )
            crop.rio.to_raster(save_path)
        return "success"
    except:
        return "error"

def run_safe_download():
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    df.columns = df.columns.str.strip().str.lower()
    if 'sample date' in df.columns: df.rename(columns={'sample date': 'date'}, inplace=True)

    print(f">> 무결성 검사 포함 안전 모드 가동. 대상: {len(df)}개")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one_point, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(df)):
            future.result()

if __name__ == "__main__":
    run_safe_download()
