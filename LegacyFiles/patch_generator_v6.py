import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging
from pyproj import Transformer

# --- Configuration ---
RAW_DIR = "/mnt/data_lake/EY_Satellite_Raw"
OUTPUT_DIR = "/mnt/data_lake/EY_Satellite_Patches_V2"
PATCH_SIZE = 256
TARGET_BANDS = ["B02", "B03", "B04", "B08"]
NUM_WORKERS = 6  # 안정성을 위해 워커 수 보수적 설정

# 에러 로그 설정
logging.basicConfig(
    filename='patch_gen_error.log', 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_single_date(date):
    """
    단일 날짜(Scene) 처리. 파일 손상 시 해당 Scene 또는 Patch를 스킵.
    """
    import rasterio # 워커 프로세스 내 임포트
    
    band_files = [os.path.join(RAW_DIR, f"T1_{date}_{b}.tif") for b in TARGET_BANDS]
    
    # 1. 파일 존재 확인
    if not all(os.path.exists(f) for f in band_files):
        return (0, f"Missing files for {date}")

    created_count = 0
    srcs = []
    
    try:
        # 2. 메타데이터 로드 및 좌표 변환기 생성
        with rasterio.open(band_files[0]) as src:
            width, height = src.width, src.height
            src_crs = src.crs
            src_transform = src.transform
            # UTM -> WGS84(Lat/Lon) 변환기
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

        # 3. 데이터 핸들 열기
        srcs = [rasterio.open(f) for f in band_files]
        
        x_steps = list(range(0, width, PATCH_SIZE))
        y_steps = list(range(0, height, PATCH_SIZE))
        
        for y in y_steps:
            for x in x_steps:
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                
                # 자투리 제거
                if window.width != PATCH_SIZE or window.height != PATCH_SIZE:
                    continue
                
                try:
                    # 데이터 읽기 (B02로 유효성 체크)
                    b02_data = srcs[0].read(1, window=window)
                    
                    if np.max(b02_data) == 0: # 빈 데이터 스킵
                        continue
                        
                    # 나머지 밴드 읽기
                    patch_tensor = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
                    patch_tensor[0] = b02_data
                    for i in range(1, 4):
                        patch_tensor[i] = srcs[i].read(1, window=window)
                    
                    # 좌표 변환 (Center Pixel)
                    center_x_px = x + PATCH_SIZE // 2
                    center_y_px = y + PATCH_SIZE // 2
                    
                    # 1. 이미지 픽셀 -> 투영 좌표(Easting, Northing)
                    px_x, px_y = rasterio.transform.xy(src_transform, center_y_px, center_x_px, offset='center')
                    
                    # 2. 투영 좌표 -> 위도/경도(Lat, Lon)
                    lon, lat = transformer.transform(px_x, px_y)
                    
                    # 저장
                    file_name = f"P_lat_{lat:.6f}_lon_{lon:.6f}_{date}.npy"
                    save_path = os.path.join(OUTPUT_DIR, file_name)
                    np.save(save_path, patch_tensor)
                    created_count += 1
                    
                except Exception as e:
                    # 읽기 중 에러 발생 (ZIPDecode 등) -> 해당 패치/Scene 기록 후 건너뜀
                    logging.error(f"Read Error at {date} (x={x}, y={y}): {str(e)}")
                    # 특정 위치만 깨진 경우 loop 계속, 심각하면 break 고려. 
                    # 여기서는 안전하게 continue
                    continue
        
    except Exception as e:
        # 파일 열기 실패 등 치명적 오류
        logging.error(f"Critical Error in Scene {date}: {str(e)}")
        return (0, str(e))
    
    finally:
        # 핸들 닫기
        for src in srcs:
            try: src.close()
            except: pass

    return (created_count, None)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 날짜 추출
    all_files = glob.glob(os.path.join(RAW_DIR, "*.tif"))
    dates = set()
    for f in all_files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 2:
            dates.add(parts[1])
    unique_dates = sorted(list(dates))
    
    print(f"--- [Processor V6] Safe Mode & Coord Fix ---")
    print(f"Targets: {len(unique_dates)} Scenes")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Log File: patch_gen_error.log")
    
    start_time = time.time()
    total_patches = 0
    failed_scenes = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_date, d): d for d in unique_dates}
        
        for future in tqdm(as_completed(futures), total=len(unique_dates), desc="Processing"):
            cnt, err = future.result()
            total_patches += cnt
            if err:
                failed_scenes.append(futures[future])
            
    elapsed = time.time() - start_time
    print(f"\n[Done] Generated: {total_patches} patches")
    print(f"Time: {elapsed:.2f} sec")
    
    if failed_scenes:
        print(f"\n[WARNING] {len(failed_scenes)} Scenes had critical errors.")
        print(f"List: {failed_scenes}")
    else:
        print("\n[SUCCESS] No critical scene failures.")

if __name__ == "__main__":
    main()
