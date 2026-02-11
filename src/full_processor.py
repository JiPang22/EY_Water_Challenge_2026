import os
import glob
import rasterio
from rasterio.windows import Window
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import re

DATA_DIR = "/mnt/data_lake/EY_Satellite_Raw"
PATCH_DIR = "/mnt/data_lake/EY_Satellite_Patches"
PATCH_SIZE = 256
MAX_WORKERS = os.cpu_count()

class GlobalProcessor:
    def __init__(self):
        os.makedirs(PATCH_DIR, exist_ok=True)
        self.bands = ['B02', 'B03', 'B04', 'B08']

    def process_scene(self, scene_id, date_str):
        # Interface: 4개 밴드 경로 생성
        band_paths = [os.path.join(DATA_DIR, f"{scene_id}_{date_str}_{b}.tif") for b in self.bands]
        
        if not all(os.path.exists(p) for p in band_paths):
            return 0

        try:
            with rasterio.open(band_paths[0]) as src:
                h, w = src.height, src.width
            
            count = 0
            for i in range(0, h, PATCH_SIZE):
                for j in range(0, w, PATCH_SIZE):
                    if i + PATCH_SIZE > h or j + PATCH_SIZE > w:
                        continue
                    
                    window = Window(j, i, PATCH_SIZE, PATCH_SIZE)
                    stack = []
                    for p in band_paths:
                        with rasterio.open(p) as src:
                            stack.append(src.read(1, window=window))
                    
                    patch = np.stack(stack, axis=0)
                    
                    # NoData 필터링: 픽셀 값의 최대값이 0이면 건너뜀
                    if np.max(patch) == 0:
                        continue
                        
                    save_name = f"P_{scene_id}_{date_str}_{i}_{j}.npy"
                    np.save(os.path.join(PATCH_DIR, save_name), patch)
                    count += 1
            return count
        except Exception:
            return 0

def run():
    # 1. RAW 폴더에서 모든 B02 파일을 찾아 씬 리스트(ID, Date) 확보
    all_b02 = glob.glob(os.path.join(DATA_DIR, "*_B02.tif"))
    scenes = []
    for path in all_b02:
        fname = os.path.basename(path)
        # 파일명 구조: {ID}_{DATE}_B02.tif -> T1_20230102_B02.tif
        match = re.match(r"(.+)_(\d{8})_B02\.tif", fname)
        if match:
            scenes.append((match.group(1), match.group(2)))

    if not scenes:
        print("✘ Error: No matching .tif files found in DATA_DIR.")
        return

    processor = GlobalProcessor()
    total_patches = 0
    start_time = time.time()

    print(f"▶ Found {len(scenes)} unique scenes from file system. Processing...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(processor.process_scene, s[0], s[1]): s for s in scenes}
        
        for future in tqdm(as_completed(futures), total=len(scenes), desc="Tiling"):
            total_patches += future.result()

    print(f"\n✔ MISSION COMPLETED")
    print(f"✔ Total Patches: {total_patches}")
    print(f"✔ Elapsed Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run()
