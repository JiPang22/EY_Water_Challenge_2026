import os
import rasterio
from rasterio.windows import Window
import numpy as np
import time
from tqdm import tqdm

class PatchProcessor:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size
        self.bands = ['B02', 'B03', 'B04', 'B08']

    def process_single_scene(self, data_dir, prefix, date_str, output_dir):
        # Interface: 4개 밴드 결합 규격 강제
        band_paths = [os.path.join(data_dir, f"{prefix}_{date_str}_{b}.tif") for b in self.bands]
        
        if not all(os.path.exists(p) for p in band_paths):
            return 0

        with rasterio.open(band_paths[0]) as src:
            h, w = src.height, src.width
            meta = src.meta

        patches_count = 0
        # Dynamic Shaping: 이미지 크기에 맞춰 패치 좌표 계산
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                if i + self.patch_size > h or j + self.patch_size > w:
                    continue
                
                window = Window(j, i, self.patch_size, self.patch_size)
                stack = []
                
                for p in band_paths:
                    with rasterio.open(p) as src:
                        stack.append(src.read(1, window=window))
                
                patch_array = np.stack(stack, axis=0) # (4, 256, 256)
                
                # NoData 필터링 (모든 값이 0인 조각 제외)
                if np.max(patch_array) == 0:
                    continue
                
                save_path = os.path.join(output_dir, f"{prefix}_{date_str}_{i}_{j}.npy")
                np.save(save_path, patch_array)
                patches_count += 1
        
        return patches_count

def unit_test():
    # Unit Test: 단일 패치 생성 및 차원 검증
    print("[Unit Test] Starting...")
    proc = PatchProcessor(patch_size=256)
    test_out = "data/patches_test"
    os.makedirs(test_out, exist_ok=True)
    
    count = proc.process_single_scene("/mnt/data_lake/EY_Satellite_Raw", "T1", "20230102", test_out)
    
    if count > 0:
        sample = np.load(os.path.join(test_out, os.listdir(test_out)[0]))
        print(f"✔ Success: Created {count} patches.")
        print(f"✔ Shape: {sample.shape} (Expected: (4, 256, 256))")
    else:
        print("✘ Failed: No patches created.")

if __name__ == "__main__":
    # ETA 모니터링 기능 포함
    start = time.time()
    unit_test()
    print(f"Total Time: {time.time() - start:.2f}s")
