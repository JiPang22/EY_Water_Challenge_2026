# src/verify.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 현재 폴더(src)에서 모듈을 찾도록 설정
sys.path.append(os.getcwd())

from interp_loader import InterpWaterDataset

def main():
    # 1. 경로 설정 (현재 스크립트 위치 기준 상위 폴더)
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
    CSV_PATH = os.path.join(BASE_DIR, 'rawData', 'water_quality_training_dataset.csv')
    CHIP_DIR = os.path.join(BASE_DIR, 'rawData', 'chips')

    print(f"Dataset Path: {CSV_PATH}")

    # 2. 데이터셋 로드
    try:
        dataset = InterpWaterDataset(CSV_PATH, CHIP_DIR)
        print(f"✅ 데이터셋 로드 성공! 총 샘플 수: {len(dataset)}")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    # 3. 보간(Interp) 샘플 찾기
    sample_idx = -1
    for i in range(len(dataset)):
        if dataset.samples[i]['type'] == 'interp':
            sample_idx = i
            break

    if sample_idx == -1:
        print("⚠️ 보간된 샘플이 하나도 없습니다. 데이터나 로직을 확인하세요.")
        return

    print(f"🔍 검증할 샘플 인덱스: {sample_idx}")
    item = dataset[sample_idx]
    info = dataset.samples[sample_idx]
    img_tensor, target = item

    # 4. 시각화 및 저장 (화면 출력 X -> 파일 저장 O)
    # 텐서 -> 이미지 변환 함수
    def to_img(t):
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        return t[:3, :, :].transpose(1, 2, 0) # (C, H, W) -> (H, W, C)

    # 데이터 로드 (시각화용 raw data)
    prev_raw = dataset._load_chip_tensor(info['prev'])
    next_raw = dataset._load_chip_tensor(info['next'])

    # 그래프 그리기
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(to_img(prev_raw))
    axes[0].set_title(f"Prev ({info['prev']['date'].date()})")

    axes[1].imshow(to_img(img_tensor))
    axes[1].set_title(f"Interpolated (alpha={info['alpha']:.2f})")

    axes[2].imshow(to_img(next_raw))
    axes[2].set_title(f"Next ({info['next']['date'].date()})")

    for ax in axes: ax.axis('off')

    # 결과 저장
    save_path = os.path.join(BASE_DIR, 'interpolation_check.png')
    plt.savefig(save_path)
    print(f"📸 검증 이미지 저장 완료: {save_path}")
    print(f"📊 로그 변환된 타겟 값: {target.numpy()}")

if __name__ == "__main__":
    main()
