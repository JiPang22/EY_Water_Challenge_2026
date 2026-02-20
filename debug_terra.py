import xarray as xr
import pandas as pd
import pystac_client
import planetary_computer
import os
import sys

# 0. 라이브러리 체크
print("🔍 Checking libraries...")
try:
    import zarr
    import adlfs
    print("✅ Libraries (zarr, adlfs) are installed.")
except ImportError as e:
    print(f"❌ Missing Library: {e}")
    print("👉 Run: pip install zarr adlfs fsspec")
    sys.exit(1)

# 1. 템플릿 로드 확인
print("\n📂 Loading submission_template.csv...")
if os.path.exists("submission_template.csv"):
    df = pd.read_csv("submission_template.csv")
    row = df.iloc[0] # 첫 번째 행만 테스트
    print(f"✅ Template loaded. Testing first row: {row['Sample Date']}")
else:
    print("❌ submission_template.csv not found!")
    sys.exit(1)

# 2. Planetary Computer 접속
print("\n🔌 Connecting to Planetary Computer API...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)
print("✅ Catalog Connected.")

# 3. TerraClimate 컬렉션 및 Asset 확보
print("\n🌍 Getting TerraClimate Asset...")
collection = catalog.get_collection("terraclimate")
asset = collection.assets["zarr-https"]
print(f"✅ Asset Found: {asset.href}")

# 4. Zarr 데이터셋 열기 (가장 에러 많이 나는 곳)
print("\n🔓 Opening Zarr Dataset (Authentication Check)...")
try:
    ds = xr.open_dataset(
        asset.href,
        engine="zarr",
        storage_options=planetary_computer.storage_options(asset.href)
    )
    print("✅ Dataset Opened Successfully!")
    print(ds)
except Exception as e:
    print(f"❌ Failed to open dataset: {e}")
    sys.exit(1)

# 5. 데이터 슬라이싱 테스트 (날짜/좌표)
print("\n🔪 Slicing Data (Date/Lat/Lon Check)...")
try:
    # 날짜 파싱
    date_str = row['Sample Date']
    print(f"   Target Date String: {date_str}")
    
    # 공백 제거 및 포맷팅 (Landsat과 동일한 잠재적 문제 체크)
    target_date = pd.to_datetime(date_str, dayfirst=True)
    print(f"   Parsed Date: {target_date}")

    lat = row['Latitude']
    lon = row['Longitude']
    
    # 데이터 추출
    print(f"   Querying: Lat={lat}, Lon={lon}, Time={target_date}")
    
    # method='nearest'로 근접 월 데이터 찾기
    subset = ds['pet'].sel(
        time=target_date, 
        lat=lat, 
        lon=lon, 
        method='nearest'
    )
    
    val = subset.values
    print(f"\n🎉 Success! Extracted Value (pet): {val}")
    
except Exception as e:
    print(f"\n❌ Error during slicing: {e}")
    sys.exit(1)
