import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import traceback
from pyproj import Transformer

# --- 설정 ---
RAW_DIR = "/mnt/data_lake/EY_Satellite_Raw"
OUTPUT_DIR = "/mnt/data_lake/EY_Satellite_Patches_V2"
PATCH_SIZE = 256
TARGET_BANDS = ["B02", "B03", "B04", "B08"]
NUM_WORKERS = 10 

def process_single_date(date):
    # 워커 프로세스 내에서 라이브러리 다시 로드 (안전장치)
    import rasterio
    import numpy as np
    
    band_files = [os.path.join(RAW_DIR, f"T1_{date}_{b}.tif") for b in TARGET_BANDS]
    
    # 1. 파일 존재 확인
    if not all(os.path.exists(f) for f in band_files):
        return (0, f"Missing files for {date}")

    created_count = 0
    
    try:
        # 2. 메타데이터 및 좌표 변환기 준비
        with rasterio.open(band_files[0]) as src:
            width, height = src.width, src.height
            src_crs = src.crs
            src_transform = src.transform
            
            # [핵심] UTM(Meters) -> Lat/Lon(WGS84) 변환기 생성
            # Zone 52N 등 로컬 좌표계를 경위도로 변경
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

        # IO 최적화를 위해 데이터셋 열기
        srcs = [rasterio.open(f) for f in band_files]
        
        # 3. 슬라이딩 윈도우
        x_steps = list(range(0, width, PATCH_SIZE))
        y_steps = list(range(0, height, PATCH_SIZE))
        
        for y in y_steps:
            for x in x_steps:
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                
                # 가장자리 스킵
                if window.width != PATCH_SIZE or window.height != PATCH_SIZE:
                    continue
                
                # 데이터 읽기 (B02 먼저)
                b02_data = srcs[0].read(1, window=window)
                if np.max(b02_data) == 0: # 빈 데이터 스킵
                    continue
                
                # 나머지 밴드 읽기
                patch_tensor = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
                patch_tensor[0] = b02_data
                for i in range(1, 4):
                    patch_tensor[i] = srcs[i].read(1, window=window)
                
                # 4. 좌표 계산 및 변환
                center_x_px = x + PATCH_SIZE // 2
                center_y_px = y + PATCH_SIZE // 2
                
                # 이미지상 좌표 (x, y) -> 투영 좌표 (Easting, Northing)
                px_x, px_y = rasterio.transform.xy(src_transform, center_y_px, center_x_px, offset='center')
                
                # 투영 좌표 -> 위도/경도 변환
                lon, lat = transformer.transform(px_x, px_y)
                
                # 5. 저장
                file_name = f"P_lat_{lat:.6f}_lon_{lon:.6f}_{date}.npy"
                save_path = os.path.join(OUTPUT_DIR, file_name)
                np.save(save_path, patch_tensor)
                created_count += 1
        
        # 리소스 해제
        for src in srcs:
            src.close()
            
    except Exception as e:
        # 에러 발생 시 상세 로그 반환
        return (0, traceback.format_exc())

    return (created_count, None)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 파일 목록 가져오기
    all_files = glob.glob(os.path.join(RAW_DIR, "*.tif"))
    dates = set()
    for f in all_files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 2:
            dates.add(parts[1])
    unique_dates = sorted(list(dates))
    
    print(f"--- [Processor V5] Coordinate Correction Enabled ---")
    print(f"Targets: {len(unique_dates)} Scenes")
    print(f"Workers: {NUM_WORKERS}")
    
    start_time = time.time()
    total_patches = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_date, d): d for d in unique_dates}
        
        for future in tqdm(as_completed(futures), total=len(unique_dates), desc="Processing"):
            cnt, err = future.result()
            total_patches += cnt
            if err:
                errors.append(err)
            
    elapsed = time.time() - start_time
    print(f"\n[Done] Generated: {total_patches} patches")
    print(f"Time: {elapsed:.2f} sec")
    
    if errors:
        print("\n!!! ERRORS DETECTED !!!")
        print(errors[0]) # 첫 번째 에러만 출력

if __name__ == "__main__":
    main()
