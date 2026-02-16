import os
import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import tempfile

# 1. 원자적 저장 함수 (Atomic Write)
def safe_save(path, data, is_json=False):
    temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    try:
        if is_json:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f)
        else:
            with os.fdopen(temp_fd, 'wb') as f:
                np.save(f, data)
        os.replace(temp_path, path) # 쓰기 완료 후 교체
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise e

def download_location(loc_data):
    lat, lon, output_base = loc_data
    loc_id = f"{lat:.4f}_{lon:.4f}"
    loc_dir = os.path.join(output_base, f"loc_{loc_id}")
    os.makedirs(loc_dir, exist_ok=True)

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    search = catalog.search(
        collections=["landsat-c2-l2", "sentinel-2-l2a"],
        bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01],
        datetime="2011-01-01/2015-12-31"
    )
    
    for item in search.item_collection():
        date = item.properties["datetime"][:10]
        npy_path = os.path.join(loc_dir, f"{date}.npy")
        json_path = os.path.join(loc_dir, f"{date}_meta.json")

        if os.path.exists(npy_path): continue # 이미 있는 건 건너뜀

        try:
            ds = stac_load([item], bands=["red", "green", "blue", "nir08", "swir16"], 
                           bbox=[lon-0.002, lat-0.002, lon+0.002, lat+0.002], resolution=10).compute()
            
            img_data = ds.to_array().values[:, 0, :32, :32]
            if img_data.size != 5120: continue

            # 안전하게 저장
            safe_save(npy_path, img_data)
            safe_save(json_path, {"cloud": item.properties.get("eo:cloud_cover", 100), "platform": item.properties.get("platform")}, is_json=True)
        except:
            continue

def run():
    df = pd.read_csv('rawData/water_quality_training_dataset.csv')
    locations = df[['Latitude', 'Longitude']].drop_duplicates().values
    
    print(f"🚀 {len(locations)}개 지점 재수집 시작 (Thread: 10)")
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(lambda x: download_location((*x, 'rawData/chips')), locations), total=len(locations)))

if __name__ == "__main__":
    run()