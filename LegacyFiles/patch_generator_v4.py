import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

RAW_DIR = "/mnt/data_lake/EY_Satellite_Raw"
OUTPUT_DIR = "/mnt/data_lake/EY_Satellite_Patches_V2"
PATCH_SIZE = 256
TARGET_BANDS = ["B02", "B03", "B04", "B08"]
NUM_WORKERS = 10 

def get_unique_dates(file_list):
    dates = set()
    for f in file_list:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 2:
            dates.add(parts[1])
    return sorted(list(dates))

def process_single_date(date):
    band_files = [os.path.join(RAW_DIR, f"T1_{date}_{b}.tif") for b in TARGET_BANDS]
    
    if not all(os.path.exists(f) for f in band_files):
        return 0

    created_count = 0
    
    try:
        with rasterio.open(band_files[0]) as src:
            width, height = src.width, src.height
            transform = src.transform
            
        x_steps = list(range(0, width, PATCH_SIZE))
        y_steps = list(range(0, height, PATCH_SIZE))
        
        srcs = [rasterio.open(f) for f in band_files]
        
        for y in y_steps:
            for x in x_steps:
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                
                if window.width != PATCH_SIZE or window.height != PATCH_SIZE:
                    continue
                
                b02_data = srcs[0].read(1, window=window)
                if np.max(b02_data) == 0:
                    continue
                
                patch_tensor = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
                patch_tensor[0] = b02_data
                for i in range(1, 4):
                    patch_tensor[i] = srcs[i].read(1, window=window)
                
                center_x_px = x + PATCH_SIZE // 2
                center_y_px = y + PATCH_SIZE // 2
                lon, lat = rasterio.transform.xy(transform, center_y_px, center_x_px, offset='center')
                
                file_name = f"P_lat_{lat:.6f}_lon_{lon:.6f}_{date}.npy"
                save_path = os.path.join(OUTPUT_DIR, file_name)
                np.save(save_path, patch_tensor)
                created_count += 1
        
        for src in srcs:
            src.close()
            
    except Exception:
        return 0

    return created_count

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_files = glob.glob(os.path.join(RAW_DIR, "*.tif"))
    unique_dates = get_unique_dates(all_files)
    
    print(f"--- [Processor V4] Multi-Core Optimization ---")
    print(f"Target Dates: {len(unique_dates)}")
    print(f"Workers: {NUM_WORKERS}")
    
    start_time = time.time()
    total_patches = 0
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_date, date): date for date in unique_dates}
        
        for future in tqdm(as_completed(futures), total=len(unique_dates), desc="Processing Scenes"):
            cnt = future.result()
            total_patches += cnt
            
    elapsed = time.time() - start_time
    print(f"\n[Done] Total Patches: {total_patches}")
    print(f"Total Time: {elapsed:.2f} sec")

if __name__ == "__main__":
    main()
