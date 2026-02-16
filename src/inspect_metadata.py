import os
import glob
import json
import numpy as np

# 경로 설정
BASE_DIR = ".."
CHIP_DIR = os.path.join(BASE_DIR, 'rawData', 'chips')

def inspect_physical_properties():
    # 1. 샘플 메타데이터 파일 찾기
    meta_files = glob.glob(f"{CHIP_DIR}/loc_*/*.json")
    
    if not meta_files:
        print("❌ 메타데이터(.json) 파일을 찾을 수 없습니다.")
        return

    # 첫 번째 파일 로드
    target_file = meta_files[0]
    with open(target_file, 'r') as f:
        meta = json.load(f)

    print(f"📂 분석 대상: {os.path.basename(target_file)}")
    print("-" * 40)
    print(" [물리적 메타데이터 키(Key) 목록] ")
    for k, v in meta.items():
        print(f" - {k}: {v}")
    print("-" * 40)

    # 2. 밴드 통계 (값의 범위 확인 -> 반사율 단위 추정)
    npy_file = target_file.replace('_meta.json', '.npy')
    if os.path.exists(npy_file):
        img = np.load(npy_file)
        print(f" [이미지 텐서 통계] Shape: {img.shape}")
        print(f" - Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.2f}")
        if img.max() > 1:
            print(" 👉 Raw DN 값(0~10000 또는 65535)입니다. -> 0~1 정규화 필요")
        else:
            print(" 👉 이미 Reflectance(0~1)로 변환된 값입니다.")

if __name__ == "__main__":
    inspect_physical_properties()
