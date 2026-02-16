import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------------
# 1. 경로 자동 설정 (이 부분이 핵심입니다)
# --------------------------------------------------------
# 현재 파일(inference.py)의 위치: .../EY_Water_Challenge_2026/src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 폴더: .../EY_Water_Challenge_2026
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 모듈을 찾을 수 있도록 src 폴더를 패스에 추가
sys.path.append(CURRENT_DIR)

# 이제 src 내부 모듈들을 안심하고 임포트
from model import MobileViT_XXS
from constants import *
from interp_loader import InterpWaterDataset

# --------------------------------------------------------
# 2. 설정값 정의
# --------------------------------------------------------
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 파일 경로를 절대 경로로 고정 (어디서 실행하든 상관없게)
# [중요] 사용자가 다운받은 템플릿 파일명: submission_template.csv
SUBMISSION_FILE = os.path.join(ROOT_DIR, "submission_template.csv")
CHIPS_DIR = os.path.join(ROOT_DIR, "rawData", "chips")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pth")
OUTPUT_FILE = os.path.join(ROOT_DIR, "final_submission.csv")

# --------------------------------------------------------
# 3. 테스트용 데이터셋 (정답지 무시)
# --------------------------------------------------------
class TestDataset(InterpWaterDataset):
    def __init__(self, csv_path, chips_dir):
        # 부모 클래스의 init을 호출하여 매칭 로직을 그대로 사용
        # mode='test'로 설정하면 정답(y)을 반환하지 않음
        super().__init__(csv_path, chips_dir, mode='test')

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 보간(Interp) 또는 단일(Single) 이미지 로드
        if item['type'] == 'interp':
            img_prev = self._load_chip_tensor(item['prev'])
            img_next = self._load_chip_tensor(item['next'])
            img_tensor = (1 - item['alpha']) * img_prev + item['alpha'] * img_next
        else:
            img_tensor = self._load_chip_tensor(item['chip'])

        return img_tensor

# --------------------------------------------------------
# 4. 추론 실행 함수
# --------------------------------------------------------
def main():
    print("="*60)
    print(f"[Inference] 시작 (Device: {DEVICE})")
    print(f"[Check] 루트 경로: {ROOT_DIR}")
    print(f"[Check] 입력 파일: {SUBMISSION_FILE}")
    print(f"[Check] 모델 파일: {MODEL_PATH}")
    print("="*60)

    # 1. 파일 존재 확인
    if not os.path.exists(SUBMISSION_FILE):
        print(f"❌ 오류: 입력 파일이 없습니다!\n   -> 경로: {SUBMISSION_FILE}")
        print("   -> 팁: 다운받은 'submission_template.csv' 파일을 프로젝트 최상위 폴더에 넣어주세요.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 학습된 모델 파일이 없습니다!\n   -> 경로: {MODEL_PATH}")
        print("   -> 팁: 먼저 'python src/train.py'를 실행해서 모델을 학습시켜주세요.")
        return

    # 2. 데이터셋 & 로더 준비
    # interp_loader가 내부적으로 'Latitude', 'Longitude', 'Sample Date'를 읽어서
    # 해당하는 위성 이미지를 찾아냅니다. (값은 무시하고 위치 정보만 사용)
    try:
        test_dataset = TestDataset(SUBMISSION_FILE, CHIPS_DIR)
    except KeyError as e:
        print(f"❌ CSV 컬럼 오류: {e}")
        print("   -> 팁: CSV 파일에 'Latitude', 'Longitude', 'Sample Date' 컬럼이 정확히 있는지 확인하세요.")
        return
    except Exception as e:
        print(f"❌ 데이터 로드 중 알 수 없는 오류: {e}")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"[Data] 총 {len(test_dataset)}개의 샘플을 처리합니다.")

    # 3. 모델 로드 (MobileViT-XXS)
    model = MobileViT_XXS(input_channels=5, output_dim=3, img_size=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("[Model] 가중치 로드 완료.")

    # 4. 예측 루프 (Prediction Loop)
    all_preds = []
    print("[Run] 예측 수행 중...")

    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Processing"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)

            # 후처리: Log Scale -> 원래 값 복원 (exp(x) - 1)
            # 학습할 때 log1p를 썼으므로, 예측할 땐 expm1로 되돌려야 함
            preds = torch.expm1(outputs).cpu().numpy()

            # 음수 방지 (수질 지표는 0보다 작을 수 없음)
            preds = np.maximum(preds, 0)

            all_preds.append(preds)

    # 전체 예측값을 하나로 합침
    final_preds = np.vstack(all_preds)

    # 5. 결과 저장 (Submission 생성)
    # 원본 템플릿을 읽어서 정답 칸만 채워 넣습니다.
    df_submission = pd.read_csv(SUBMISSION_FILE)

    # 데이터 개수 검증 (만약 위성 이미지를 못 찾은 샘플이 있다면 개수가 다를 수 있음)
    n_rows = min(len(df_submission), len(final_preds))

    if len(df_submission) != len(final_preds):
        print(f"⚠️ 경고: 원본({len(df_submission)}개)과 예측({len(final_preds)}개)의 개수가 다릅니다.")
        print(f"   -> 앞부분 {n_rows}개만 채워서 저장합니다.")

    # 예측값 채우기
    # TARGET_COLS 순서: ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    df_submission.iloc[:n_rows, df_submission.columns.get_indexer(TARGET_COLS)] = final_preds[:n_rows]

    # CSV 저장
    df_submission.to_csv(OUTPUT_FILE, index=False)

    print("="*60)
    print(f"✅ [완료] 최종 제출 파일 생성됨: {OUTPUT_FILE}")
    print("="*60)
    print(df_submission.head())

if __name__ == "__main__":
    main()
