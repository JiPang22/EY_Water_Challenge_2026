import pandas as pd
import pystac_client
import planetary_computer as pc
import rioxarray
import os
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

# 인터페이스 명세
TRAIN_CSV = "data/raw/water_quality_training_dataset.csv"
SAVE_DIR = "data/raw/sample_chips/"
CHIP_SIZE = 32

def download_and_visualize():
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    df.columns = df.columns.str.strip().str.lower()
    if 'sample date' in df.columns: df.rename(columns={'sample date': 'date'}, inplace=True)

    samples = df.head(5)
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    for idx, row in samples.iterrows():
        lat, lon = row['latitude'], row['longitude']
        row_id = row.get('id', idx)
        date = pd.to_datetime(row['date'])
        date_range = f"{date.year}-01-01/{date.year}-12-31"

        print(f">> [ID {row_id}] 검색: Lat {lat}, Lon {lon}")

        try:
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01],
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": 50}}
            )
            items = search.item_collection()
            if not items: continue

            item = pc.sign(sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[0])
            rgb_bands = ["red", "green", "blue"]
            data_list = []

            for asset in rgb_bands:
                ds = rioxarray.open_rasterio(item.assets[asset].href)

                # 1. 좌표 변환: WGS84(Lat/Lon) -> 위성 좌표계(UTM 등)
                # $$ (lon, lat) \xrightarrow{Transformer} (x_{utm}, y_{utm}) $$_
                transformer = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
                x_utm, y_utm = transformer.transform(lon, lat)

                # 2. .sel()을 사용하여 가장 가까운 픽셀 지점으로 이동
                # 인덱스 에러 방지를 위해 중심점 주변을 슬라이싱
                center_ds = ds.sel(x=x_utm, y=y_utm, method="nearest")

                # 3. 중심점 기준 32x32 크롭 (numpy 인덱싱 활용)
                x_idx = np.where(ds.x == center_ds.x)[0][0]
                y_idx = np.where(ds.y == center_ds.y)[0][0]

                crop = ds.isel(
                    x=slice(max(0, x_idx - CHIP_SIZE // 2), x_idx + CHIP_SIZE // 2),
                    y=slice(max(0, y_idx - CHIP_SIZE // 2), y_idx + CHIP_SIZE // 2)
                )

                # 4. 시각화용 정규화
                arr = crop.values[0].astype(float)
                arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-5)
                data_list.append(arr)

            # RGB 스택 생성
            rgb_img = np.stack(data_list, axis=-1)

            plt.figure(figsize=(5,5))
            plt.imshow(rgb_img)
            plt.title(f"ID: {row_id} | Coords: {lat}, {lon}")
            plt.axis('off')
            plt.savefig(f"{SAVE_DIR}verify_{row_id}.png")
            plt.close()
            print(f">> [ID {row_id}] 시각화 성공: {SAVE_DIR}verify_{row_id}.png")

        except Exception as e:
            print(f"!! [ID {row_id}] 에러: {e}")

if __name__ == "__main__":
    download_and_visualize()
