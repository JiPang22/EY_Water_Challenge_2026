import os
import rasterio
import numpy as np
from PIL import Image
import time

def scale_band(band):
    # 2% ~ 98% Percentile Clipping으로 선명도 확보
    # 수식: $$ output = \frac{input - v_{min}}{v_{max} - v_{min}} \times 255 $$
    v_min, v_max = np.percentile(band, [2, 98])
    if v_max == v_min:
        return np.zeros_like(band, dtype=np.uint8)
    scaled = (band.astype(np.float32) - v_min) * 255 / (v_max - v_min)
    return np.clip(scaled, 0, 255).astype(np.uint8)

def create_preview(data_dir, prefix, date_str, output_path):
    # Interface: B04(R), B03(G), B02(B) 순서 강제
    bands = ['B04', 'B03', 'B02']
    rgb_data = []
    
    start_time = time.time()
    console_prefix = f"[{prefix}_{date_str}]"
    
    try:
        for band in bands:
            path = os.path.join(data_dir, f"{prefix}_{date_str}_{band}.tif")
            # Unit Test: 파일 존재 여부 확인
            if not os.path.exists(path):
                print(f"{console_prefix} ✘ Error: {path} not found")
                return
                
            with rasterio.open(path) as src:
                # Dynamic Shaping: 원본 크기 그대로 로드
                rgb_data.append(scale_band(src.read(1)))
        
        # RGB 스택 (H, W, 3) 및 저장
        rgb_img = np.stack(rgb_data, axis=-1)
        Image.fromarray(rgb_img).save(output_path)
        
        elapsed = time.time() - start_time
        print(f"{console_prefix} ✔ Preview saved: {output_path} ({elapsed:.2f}s)")
        
    except Exception as e:
        print(f"{console_prefix} ✘ Exception: {str(e)}")

if __name__ == "__main__":
    # gdalinfo에서 확인된 첫 번째 파일 기준으로 테스트
    DATA_PATH = "/mnt/data_lake/EY_Satellite_Raw"
    create_preview(DATA_PATH, "T1", "20230102", "preview_T1_20230102.png")
