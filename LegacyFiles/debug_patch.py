import os
import glob
import rasterio
import numpy as np
import traceback

RAW_DIR = "/mnt/data_lake/EY_Satellite_Raw"
TARGET_BANDS = ["B02", "B03", "B04", "B08"]

print("--- [DEBUG START] ---")

# 1. 파일 스캔 확인
files = glob.glob(os.path.join(RAW_DIR, "*.tif"))
if not files:
    print(f"FATAL: No .tif files found in {RAW_DIR}")
    exit(1)

# 2. 첫 번째 날짜 추출 테스트
sample_file = os.path.basename(files[0])
print(f"Sample File Found: {sample_file}")
try:
    date = sample_file.split('_')[1]
    print(f"Extracted Date: {date}")
except IndexError:
    print("FATAL: Filename parsing failed.")
    exit(1)

# 3. 밴드 파일 경로 검증
band_files = [os.path.join(RAW_DIR, f"T1_{date}_{b}.tif") for b in TARGET_BANDS]
print("Checking Band Files:")
for f in band_files:
    exists = os.path.exists(f)
    print(f"  - {os.path.basename(f)}: {'[OK]' if exists else '[MISSING]'}")
    if not exists:
        print("FATAL: Missing required band file.")
        exit(1)

# 4. Rasterio 읽기 테스트 (에러 발생 구간 예측)
print("\nAttempting to read data via Rasterio...")
try:
    with rasterio.open(band_files[0]) as src:
        print(f"  metadata: {src.meta}")
        print(f"  shape: {src.width} x {src.height}")
        
        # 중앙부 256x256 읽기 시도
        win = rasterio.windows.Window(src.width//2, src.height//2, 256, 256)
        data = src.read(1, window=win)
        
        print(f"  Read Success! Data Max Value: {np.max(data)}")
        if np.max(data) == 0:
            print("  WARNING: Data is all zeros (Black Image)")
        
except Exception:
    print("\n!!! ERROR CAUGHT !!!")
    traceback.print_exc()

print("--- [DEBUG END] ---")
