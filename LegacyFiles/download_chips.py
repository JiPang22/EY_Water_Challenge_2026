import pandas as pd
import pystac_client
import planetary_computer as pc
import rioxarray
import os
from tqdm import tqdm

# 인터페이스 명세
TRAIN_CSV = "data/raw/water_quality_training_dataset.csv"
SAVE_DIR = "data/raw/satellite_chips/"
CHIP_SIZE = 32

def download_landsat_chips():
    if not os.path.exists(TRAIN_CSV):
        print("Error: train 데이터가 없습니다.")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)

    # 컬럼명 표준화 (KeyError 방지)
    df.columns = df.columns.str.strip().str.lower()
    if 'sample date' in df.columns: df.rename(columns={'sample date': 'date'}, inplace=True)

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    print(f">> {len(df)}개 지점 이미지 추출 시작...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon = row['latitude'], row['longitude']
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
            if not items: continue

            item = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[0]
            signed_item = pc.sign(item)

            # 1등 전략: 물 분석에 필수적인 5개 밴드 추출
            assets = ["red", "green", "blue", "nir08", "swir16"]

            for asset_key in assets:
                try:
                    ds = rioxarray.open_rasterio(signed_item.assets[asset_key].href)
                    y, x = ds.rio.idx(lon, lat)
                    crop = ds.isel(x=slice(x-CHIP_SIZE//2, x+CHIP_SIZE//2),
                                   y=slice(y-CHIP_SIZE//2, y+CHIP_SIZE//2))

                    output_fn = f"{SAVE_DIR}id_{row['id']}_{asset_key}.tif"
                    crop.rio.to_raster(output_fn)
                except: continue
        except: continue

    print(f">> 완료: {SAVE_DIR}")

if __name__ == "__main__":
    download_landsat_chips()
