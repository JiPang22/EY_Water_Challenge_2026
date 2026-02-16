# file_name: patch_generator_v3.py
import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from datetime import datetime
import time

# --- Configuration ---
RAW_DIR = "/mnt/data_lake/EY_Satellite_Raw"
OUTPUT_DIR = "/mnt/data_lake/EY_Satellite_Patches_V2"
PATCH_SIZE = 256
TARGET_BANDS = ["B02", "B03", "B04", "B08"] # Blue, Green, Red, NIR

def get_unique_dates(file_list):
    # 파일명 예시: T1_20230102_B02.tif -> 20230102 추출
    dates = set()
    for f in file_list:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 2:
            dates.add(parts[1])
    return sorted(list(dates))

def process_patches():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 원본 파일 스캔
    all_files = glob.glob(os.path.join(RAW_DIR, "*.tif"))
    unique_dates = get_unique_dates(all_files)
    
    print(f"[INFO] 총 {len(unique_dates)}개의 날짜(Scene)를 처리합니다.")
    print(f"[INFO] 대상 밴드: {TARGET_BANDS}")

    start_time = time.time()

    # 2. 날짜별 처리 루프
    for date_idx, date in enumerate(unique_dates):
        # 해당 날짜의 밴드 파일 경로 구성
        band_files = [os.path.join(RAW_DIR, f"T1_{date}_{b}.tif") for b in TARGET_BANDS]
        
        # 4개 밴드가 모두 존재하는지 확인
        if not all(os.path.exists(f) for f in band_files):
            print(f"[WARN] {date} 데이터 손상 (밴드 누락). 건너뜁니다.")
            continue

        # 기준 파일(B02)을 열어 메타데이터 획득
        with rasterio.open(band_files[0]) as src:
            meta = src.meta
            width, height = src.width, src.height
            transform = src.transform
            
            # Sliding Window 계산
            # x, y 단계별로 루프
            x_steps = list(range(0, width, PATCH_SIZE))
            y_steps = list(range(0, height, PATCH_SIZE))
            
            total_patches = len(x_steps) * len(y_steps)
            
            # 진행률 표시 (ETA 포함)
            desc = f"Processing {date} ({date_idx + 1}/{len(unique_dates)})"
            with tqdm(total=total_patches, desc=desc, unit="patch") as pbar:
                
                for y in y_steps:
                    for x in x_steps:
                        # 윈도우 정의
                        window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                        
                        # 가장자리(자투리) 제외
                        if window.width != PATCH_SIZE or window.height != PATCH_SIZE:
                            pbar.update(1)
                            continue

                        # 3. 좌표 계산 (핵심)
                        # 윈도우 중앙 픽셀의 지도 좌표(Lat/Lon) 추출
                        center_x_px = x + PATCH_SIZE // 2
                        center_y_px = y + PATCH_SIZE // 2
                        lon, lat = rasterio.transform.xy(transform, center_y_px, center_x_px, offset='center')
                        
                        # 4. 데이터 스택킹 (4, 256, 256)
                        patch_tensor = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
                        
                        try:
                            for b_idx, b_path in enumerate(band_files):
                                with rasterio.open(b_path) as b_src:
                                    patch_tensor[b_idx] = b_src.read(1, window=window)
                        except Exception as e:
                            # 읽기 오류 시 건너뜀
                            continue

                        # 데이터 유효성 체크 (모두 0이면 스킵 - 바다/노데이터)
                        if np.max(patch_tensor) == 0:
                            pbar.update(1)
                            continue

                        # 5. 저장 (파일명에 좌표와 날짜 포함)
                        # Format: P_lat_{lat}_lon_{lon}_{date}.npy
                        file_name = f"P_lat_{lat:.6f}_lon_{lon:.6f}_{date}.npy"
                        save_path = os.path.join(OUTPUT_DIR, file_name)
                        np.save(save_path, patch_tensor)
                        
                        pbar.update(1)
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] 모든 처리가 완료되었습니다.")
    print(f"소요 시간: {elapsed:.2f}초")
    print(f"저장 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_patches()
