import pandas as pd
import pystac_client
import planetary_computer as pc
import rioxarray
import os
import numpy as np
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 인터페이스 명세
TRAIN_CSV = "data/raw/water_quality_training_dataset.csv"
SAVE_DIR = "data/raw/satellite_chips/"
CHIP_SIZE = 32
MAX_WORKERS = 32 # 32스레드 설정

def download_one_point(row):
    """단일 지점의 5개 밴드를 다운로드하는 함수"""
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    lat, lon = row['latitude'], row['longitude']
    row_id = row.get('id', row.name)
    date = pd.to_datetime(row['date'])
    date_range = f"{date.year}-01-01/{date.year}-12-31"

    try:
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=[lon-0.005, lat-0.005, lon+0.005, lat+0.005],
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 30}}
        )
        items = search.item_collection()
        if not items: return False
        
        item = pc.sign(sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[0])
        assets = ["red", "green", "blue", "nir08", "swir16"]
        
        for asset in assets:
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
            crop.rio.to_raster(f"{SAVE_DIR}id_{row_id}_{asset}.tif")
        return True
    except:
        return False

def run_fast_download():
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    df.columns = df.columns.str.strip().str.lower()
    if 'sample date' in df.columns: df.rename(columns={'sample date': 'date'}, inplace=True)

    print(f">> {MAX_WORKERS} 스레드 가동. 대상: {len(df)}개 지점")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one_point, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(df)):
            future.result()

if __name__ == "__main__":
    run_fast_download()
