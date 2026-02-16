import pystac_client
import planetary_computer
import requests
from tqdm import tqdm
import os

print(">>> [1/3] MPC 서버 접속 및 검색 중...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# 워싱턴 DC 근처, 2023년 1월
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[-77.01, 38.89, -76.99, 38.91],
    datetime="2023-01-01/2023-01-31",
    query={"eo:cloud_cover": {"lt": 10}}
)

items = search.item_collection()

if not items:
    print("❌ 검색 결과 없음.")
    exit()

print(f">>> [2/3] 검색 완료: {len(items)}장 발견")
item = items[0]
image_url = item.assets["B04"].href

print(f">>> [3/3] 다운로드 시작: {item.id}")

# 스트리밍 모드로 다운로드 (진행률 표시)
response = requests.get(image_url, stream=True)
total_size = int(response.headers.get('content-length', 0))

filename = "test_image_speed_check.tif"

with open(filename, "wb") as f, tqdm(
    desc=filename,
    total=total_size,
    unit='iB',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024):
        size = f.write(data)
        bar.update(size)

print("\n✅ 다운로드 완료.")
