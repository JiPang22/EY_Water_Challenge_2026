import pystac_client
import planetary_computer
import requests
import time

print(">>> MPC 접속 시도...")
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
print(f">>> 검색된 이미지: {len(items)}장")

if len(items) > 0:
    item = items[0]
    print(f">>> 다운로드: {item.id}")
    
    # B04 밴드 다운로드
    image_url = item.assets["B04"].href
    response = requests.get(image_url)
    
    with open("test_image_B04.tif", "wb") as f:
        f.write(response.content)
    print("✅ 다운로드 완료: test_image_B04.tif")
else:
    print("❌ 이미지 없음")
