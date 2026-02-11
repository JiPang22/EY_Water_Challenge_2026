import tifffile as tiff
import os

# 아무 파일이나 하나 지정
sample_path = 'data/raw/satellite_chips/id_0_red.tif'
if os.path.exists(sample_path):
    img = tiff.imread(sample_path)
    print(f"데이터 모양: {img.shape}") # (32, 32)가 나와야 함
    print(f"데이터 타입: {img.dtype}")
    print(f"픽셀 값 샘플:\n{img[:5, :5]}") # 숫자 데이터가 들어있는지 확인
