"""
[08단계] 테스트 세트 추론 및 예측 생성

- 입력:
  - submission_template.csv (또는 테스트용 CSV)
  - landsat_features_test.csv
  - terraclimate_features_test.csv
  - rawData/chips (이미지 칩)
  - models/best_multimodal_model.pth (학습된 모델 가중치)
- 출력:
  - submission.csv (예측이 채워진 최종 제출 파일)

이 파일은 기존 `inference.py`를
실행 순서가 보이도록 파일명과 주석을 정리한 버전입니다.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
import re
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MobileViT_XXS_Multimodal

# ⚙️ [설정]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# 📂 [경로]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_multimodal_model.pth")
CHIPS_DIR = os.path.join(ROOT_DIR, "rawData", "chips")
OUTPUT_PATH = os.path.join(ROOT_DIR, "submission.csv")

try:
    from constants import LS8_SCALE, LS8_OFFSET, S2_SCALE
except ImportError:
    LS8_SCALE = 0.0000275
    LS8_OFFSET = -0.2
    S2_SCALE = 10000.0


def find_test_csv(root_dir: str) -> str | None:
    """
    제출 템플릿/테스트 CSV 후보들을 순서대로 탐색하여,
    가장 먼저 발견된 파일 경로를 반환합니다.

    - submission_template.csv
    - water_quality_test*.csv
    - test_dataset.csv
    """
    patterns = ["submission_template.csv", "water_quality_test*.csv", "test_dataset.csv"]
    search_dirs = [root_dir, os.path.join(root_dir, "rawData")]

    for pattern in patterns:
        for s_dir in search_dirs:
            found = glob.glob(os.path.join(s_dir, pattern))
            # training 이라는 단어가 포함된 파일은 제외
            valid = [
                f for f in found if "training" not in os.path.basename(f).lower()
            ]
            if valid:
                return valid[0]
    return None


class TestDataset(Dataset):
    """
    테스트 CSV + 외부 피처(Landsat/TerraClimate)를 이용해
    (이미지, 탭 피처, 행 인덱스)를 반환하는 PyTorch Dataset입니다.
    """

    def __init__(self, csv_path: str, chips_dir: str):
        print(f"📄 Loading Test CSV from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        # 컬럼 이름 양쪽 공백 제거
        self.df.columns = self.df.columns.str.strip()

        # 1. 외부 피처 CSV 로드 (테스트용)
        csv_dir = os.path.dirname(csv_path)
        feat_candidates = [
            (
                os.path.join(csv_dir, "landsat_features_test.csv"),
                os.path.join(csv_dir, "terraclimate_features_test.csv"),
            ),
            (
                os.path.join(ROOT_DIR, "landsat_features_test.csv"),
                os.path.join(ROOT_DIR, "terraclimate_features_test.csv"),
            ),
        ]

        landsat_df = None
        for l_path, t_path in feat_candidates:
            if os.path.exists(l_path) and os.path.exists(t_path):
                try:
                    landsat_df = pd.read_csv(l_path)
                    terra_df = pd.read_csv(t_path)
                    merge_keys = ["Latitude", "Longitude", "Sample Date"]
                    landsat_cols = merge_keys + [
                        "nir",
                        "green",
                        "swir16",
                        "swir22",
                        "NDMI",
                        "MNDWI",
                    ]

                    # 병합 시도: 기본 CSV에 Landsat/TerraClimate 피처를 붙임
                    self.df = pd.merge(
                        self.df, landsat_df[landsat_cols], on=merge_keys, how="left"
                    )
                    self.df = pd.merge(
                        self.df,
                        terra_df[merge_keys + ["pet"]],
                        on=merge_keys,
                        how="left",
                    )
                    print(
                        f"📦 Loaded external features from "
                        f"{os.path.basename(l_path)} / {os.path.basename(t_path)}"
                    )
                    break
                except Exception as e:
                    print(f"⚠️ Merge failed: {e}")
                    continue

        # 외부 피처를 전혀 찾지 못한 경우: 0으로 채운다.
        if landsat_df is None:
            print("⚠️ No external features found. Filling with zeros.")
            for col in [
                "nir",
                "green",
                "swir16",
                "swir22",
                "NDMI",
                "MNDWI",
                "pet",
            ]:
                self.df[col] = 0.0

        # 탭 피처 컬럼 정의 (학습과 동일한 순서)
        self.tab_cols = [
            "Latitude",
            "Longitude",
            "nir",
            "green",
            "swir16",
            "swir22",
            "NDMI",
            "MNDWI",
            "pet",
        ]
        self.df[self.tab_cols] = self.df[self.tab_cols].fillna(0)
        self.df["dt"] = pd.to_datetime(
            self.df["Sample Date"], dayfirst=True, errors="coerce"
        )

        # 칩 파일 메타데이터 수집
        self.chips = []
        if os.path.exists(chips_dir):
            print(f"🔍 Scanning chips in {chips_dir}...")
            npy_files = glob.glob(os.path.join(chips_dir, "**", "*.npy"), recursive=True)
            for f_path in tqdm(npy_files, desc="Parsing Chips"):
                try:
                    parent = os.path.basename(os.path.dirname(f_path))
                    f_name = os.path.basename(f_path)
                    m_loc = re.search(
                        r"loc_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)", parent
                    )
                    m_date = re.search(r"(\d{4}-\d{2}-\d{2})", f_name)
                    if m_loc and m_date:
                        self.chips.append(
                            {
                                "lat": float(m_loc.group(1)),
                                "lon": float(m_loc.group(2)),
                                "dt": datetime.strptime(
                                    m_date.group(1), "%Y-%m-%d"
                                ),
                                "path": f_path,
                            }
                        )
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.df)

    def _load_chip(self, path: str) -> torch.Tensor:
        """
        .npy 칩 파일을 읽어서 (C, H, W) 형태의 텐서로 변환합니다.
        """
        try:
            img = np.load(path).astype(np.float32)
            if img.ndim == 3 and img.shape[-1] == 5:
                img = img.transpose(2, 0, 1)
            elif img.ndim == 2:
                img = np.stack([img] * 5, axis=0)

            if np.max(img) > 20000:
                img = img / S2_SCALE
            elif np.max(img) > 1:
                img = img * LS8_SCALE + LS8_OFFSET

            return torch.from_numpy(np.clip(img, 0, 1))
        except Exception:
            return torch.zeros((5, 32, 32), dtype=torch.float32)

    def __getitem__(self, idx: int):
        """
        - 이미지 텐서
        - 탭 피처 텐서
        - 데이터프레임 행 인덱스(idx)
        를 반환합니다.
        """
        row = self.df.iloc[idx]
        pos_chips = [
            c
            for c in self.chips
            if abs(c["lat"] - row["Latitude"]) < 0.001
            and abs(c["lon"] - row["Longitude"]) < 0.001
        ]

        if pos_chips:
            # 날짜 차이가 가장 작은 칩 1개 선택
            best_chip = min(
                pos_chips,
                key=lambda x: abs((x["dt"] - row["dt"]).total_seconds()),
            )
            img = self._load_chip(best_chip["path"])
        else:
            img = torch.zeros((5, 32, 32), dtype=torch.float32)

        tabular = torch.from_numpy(row[self.tab_cols].values.astype(np.float32))

        # 행 번호(idx)를 함께 반환해, 나중에 DataFrame에 예측치를 채워 넣을 때 사용
        return img, tabular, idx


def main():
    print(f"🚀 [System] Inference on {DEVICE}")
    test_csv = find_test_csv(ROOT_DIR)
    if not test_csv:
        print("❌ Test CSV not found.")
        return

    # 1. 학습된 모델 로드
    model = MobileViT_XXS_Multimodal(
        image_channels=5, tabular_dim=9, output_dim=3
    ).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 테스트 데이터셋/로더 구성
    ds = TestDataset(test_csv, CHIPS_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 템플릿 데이터프레임 복사 (여기에 예측 결과를 채워 넣음)
    submission_df = ds.df.copy()

    target_cols = [
        "Total Alkalinity",
        "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]
    for col in target_cols:
        submission_df[col] = 0.0

    print("\n🔮 Starting Prediction...")
    results: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for images, tabular, indices in tqdm(loader, desc="Predicting"):
            images, tabular = images.to(DEVICE), tabular.to(DEVICE)
            outputs = model(images, tabular)

            # 학습 시 log1p를 썼으므로, 예측 시 expm1로 원 스케일 복원
            preds = torch.expm1(outputs)
            # 물리량 특성상 음수가 되지 않도록 클리핑
            preds = torch.clamp(preds, min=0.0)

            preds_np = preds.cpu().numpy()

            # 각 행 인덱스에 해당하는 예측값 저장
            for idx, pred in zip(indices, preds_np):
                results[int(idx.item())] = pred

    # 4. DataFrame에 예측값 채워 넣기
    print("📝 Writing results to dataframe...")
    for idx, pred in results.items():
        submission_df.loc[idx, "Total Alkalinity"] = float(pred[0])
        submission_df.loc[idx, "Electrical Conductance"] = float(pred[1])
        submission_df.loc[idx, "Dissolved Reactive Phosphorus"] = float(pred[2])

    # 5. 원본 템플릿 컬럼만 유지 + 타겟 컬럼이 없으면 뒤에 추가
    original_cols = pd.read_csv(test_csv).columns.str.strip().tolist()
    final_cols = list(original_cols)
    for col in target_cols:
        if col not in final_cols:
            final_cols.append(col)

    submission_df = submission_df[final_cols]
    submission_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n🎉 Submission saved to: {OUTPUT_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    main()

