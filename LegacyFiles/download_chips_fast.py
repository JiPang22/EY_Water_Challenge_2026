import pandas as pd
import pystac_client
import planetary_computer as pc
import rioxarray
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 인터페이스 명세
TRAIN_CSV = "data/raw/water_quality_training_dataset.csv"
SAVE_DIR = "data/raw/satellite_chips/"
CHIP_SIZE = 32
MAX_WORKERS = 32  # 요청하신 32스레드 설정

def download_one_chip(row):
    """단일 지점의 이미지 조각을 다운로드하는 함수"""
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    lat, lon, row_id = row['latitude'], row['longitude'], row['id']
    date = pd.to_datetime(row['date'])
    date_range = f"{date.year}-{date.month:02d}-01/{date.year}-{date.month:02d}-28"
    
    try:
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01],
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 30}}
        )
        
        items = search.item_collection()
        if not items: return False
            
        item = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[0]
        signed_item = pc.sign(item)
        
        assets = ["red", "green", "blue", "nir08", "swir16"]
        for asset_key in assets:
            ds = rioxarray.open_rasterio(signed_item.assets[asset_key].href)
            y, x = ds.rio.idx(lon, lat)
            crop = ds.isel(x=slice(x-CHIP_SIZE//2, x+CHIP_SIZE//2), 
                           y=slice(y-CHIP_SIZE//2, y+CHIP_SIZE//2))
            crop.rio.to_raster(f"{SAVE_DIR}id_{row_id}_{asset_key}.tif")
        return True
    except:
        return False

def fast_process():
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    df.columns = df.columns.str.strip().str.lower()
    if 'sample date' in df.columns: df.rename(columns={'sample date': 'date'}, inplace=True)

    print(f">> {MAX_WORKERS} 스레드로 병렬 다운로드 시작 (총 {len(df)}개 지점)")
    
    # 병렬 실행 (as_completed로 진행 상황 트래킹)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one_chip, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(df)):
            future.result()

if __name__ == "__main__":
    fast_process()
