import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------------
# 1. 절대 경로 설정
# --------------------------------------------------------
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from model import MobileViT_XXS
    from interp_loader import InterpWaterDataset
    try:
        from constants import TARGET_COLS
    except ImportError:
        TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
except ImportError as e:
    print(f"❌ 필수 모듈 임포트 실패: {e}")
    sys.exit(1)

# --------------------------------------------------------
# 2. 설정값
# --------------------------------------------------------
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEMPLATE_FILE = os.path.join(ROOT_DIR, "submission_template.csv")
CLEANED_FILE = os.path.join(ROOT_DIR, "cleaned_temp.csv")
OUTPUT_FILE = os.path.join(ROOT_DIR, "submission.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pth")
CHIPS_DIR = os.path.join(ROOT_DIR, "rawData", "chips")

# --------------------------------------------------------
# 3. 데이터셋 클래스 (버그 수정됨)
# --------------------------------------------------------
class TestDataset(InterpWaterDataset):
    def __init__(self, csv_path, chips_dir):
        super().__init__(csv_path, chips_dir, mode='test')

    def __getitem__(self, idx):
        item = self.samples[idx]

        # [수정] 타입별로 안전하게 로드
        if item['type'] == 'interp':
            img_prev = self._load_chip_tensor(item['prev'])
            img_next = self._load_chip_tensor(item['next'])
            img_tensor = (1 - item['alpha']) * img_prev + item['alpha'] * img_next

        elif item['type'] == 'single':
            img_tensor = self._load_chip_tensor(item['chip'])

        else: # item['type'] == 'dummy' (칩이 없을 때)
            # 5채널 32x32 빈 이미지 생성 (검은 화면)
            img_tensor = torch.zeros((5, 32, 32), dtype=torch.float32)

        return img_tensor

# --------------------------------------------------------
# 4. CSV 전처리
# --------------------------------------------------------
def prepare_csv(input_path, output_path):
    print(f"[Check] 템플릿 파일 로드: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 파일 없음: {input_path}")
        return False

    try:
        df = pd.read_csv(input_path, sep=None, engine='python')
        df.columns = df.columns.str.strip()

        required = {'latitude': 'Latitude', 'longitude': 'Longitude', 'sample date': 'Sample Date'}
        current_cols_lower = {col.lower(): col for col in df.columns}

        for req_lower, req_proper in required.items():
            if req_lower not in current_cols_lower:
                print(f"❌ 필수 컬럼 누락: {req_proper}")
                return False
            actual_col = current_cols_lower[req_lower]
            if actual_col != req_proper:
                df.rename(columns={actual_col: req_proper}, inplace=True)

        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"❌ CSV 처리 오류: {e}")
        return False

# --------------------------------------------------------
# 5. 메인 실행
# --------------------------------------------------------
def main():
    print("="*60)
    print(f"[Inference] 추론 시작 (Device: {DEVICE})")
    print("="*60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return

    if not prepare_csv(TEMPLATE_FILE, CLEANED_FILE):
        return

    # 데이터 로드
    try:
        test_dataset = TestDataset(CLEANED_FILE, CHIPS_DIR)
        # 안전장치: 샘플이 0개여도 강제로 200개 더미를 생성했을 것임
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"[Data] 처리할 샘플: {len(test_dataset)}개")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 모델 로드
    model = MobileViT_XXS(input_channels=5, output_dim=3, img_size=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("[Model] 로드 완료.")

    # 추론
    all_preds = []
    print("[Run] 예측 중...")

    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Predicting"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.expm1(outputs).cpu().numpy()
            preds = np.maximum(preds, 0)
            all_preds.append(preds)

    if not all_preds:
        print("❌ 결과 없음.")
        return

    final_preds = np.vstack(all_preds)

    # 저장
    df_submission = pd.read_csv(CLEANED_FILE)
    n_rows = min(len(df_submission), len(final_preds))

    df_submission.iloc[:n_rows, df_submission.columns.get_indexer(TARGET_COLS)] = final_preds[:n_rows]
    df_submission.to_csv(OUTPUT_FILE, index=False)

    if os.path.exists(CLEANED_FILE):
        try: os.remove(CLEANED_FILE)
        except: pass

    print("="*60)
    print(f"✅ [성공] 제출 파일 생성됨: {OUTPUT_FILE}")
    print("="*60)
    print(df_submission.head())

if __name__ == "__main__":
    main()
