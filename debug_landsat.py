import pystac_client
import planetary_computer
import pandas as pd
import os

# 1. 템플릿 로드 확인
print("Checking template...")
if os.path.exists("submission_template.csv"):
    df = pd.read_csv("submission_template.csv")
    row = df.iloc[0] # 첫 번째 행만 테스트
    print(f"✅ Template loaded. Testing first row: {row['Sample Date']}")
else:
    print("❌ submission_template.csv not found!")
    exit()

# 2. Planetary Computer 접속 테스트
print("\nConnecting to Planetary Computer...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)
print("✅ Catalog Connected.")

# 3. 데이터 검색 테스트 (여기서 에러날 확률 90%)
print("\nSearching Landsat data...")
lat = row['Latitude']
lon = row['Longitude']
date = pd.to_datetime(row['Sample Date'], dayfirst=True)

bbox = [lon-0.001, lat-0.001, lon+0.001, lat+0.001]
time_range = f"{date - pd.Timedelta(days=15)}/{date + pd.Timedelta(days=15)}"

print(f"Querying BBox: {bbox}")
print(f"Querying Time: {time_range}")

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 30}}
)
items = search.item_collection()
print(f"✅ Search Success! Found {len(items)} items.")

if len(items) > 0:
    print("\nSigning Item (Access Token Check)...")
    item = items[0]
    signed_item = planetary_computer.sign(item)
    print(f"✅ Signed Asset URL: {signed_item.assets['red'].href}")
else:
    print("⚠️ No items found. (Date or Coordinate issue)")
