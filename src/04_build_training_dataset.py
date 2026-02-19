"""
[04단계] 학습용 데이터셋(이미지 + 탭 피처) 구성

- 입력:
  - water_quality_training_dataset.csv
  - landsat_features_training.csv
  - terraclimate_features_training.csv
  - rawData/chips 아래의 위성 이미지 칩(.npy)
- 출력:
  - PyTorch Dataset 클래스: InterpWaterDataset

이 파일은 기존 `interp_loader.py`의 핵심 로직을,
실행 단계가 보이도록 파일명과 주석을 정리한 버전입니다.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime
import re
import glob
from tqdm import tqdm

try:
    from constants import LS8_SCALE, LS8_OFFSET, S2_SCALE
except ImportError:
    LS8_SCALE = 0.0000275
    LS8_OFFSET = -0.2
    S2_SCALE = 10000.0


class InterpWaterDataset(Dataset):
    """
    위성 이미지 칩 + 탭 피처 + 수질 라벨을 하나의 PyTorch Dataset으로 만드는 클래스입니다.

    - 한 행(row)은 한 번의 수질 관측을 의미합니다.
    - 각 관측에는
      - 위치: 위도(Latitude), 경도(Longitude)
      - 날짜: Sample Date
      - 타겟: (Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus)
      - 외부 피처: Landsat/테라클라이밋 지수
    - 이미지 칩은 `chips_dir` 아래의 .npy 파일에서, 이름/폴더명으로
      위도/경도/날짜를 추출해 매칭합니다.
    """

    def __init__(self, csv_path: str, chips_dir: str, mode: str = "train"):
        self.mode = mode
        self.chips_dir = chips_dir

        # 1. CSV 로드 및 외부 피처 병합
        root_path = os.path.dirname(os.path.dirname(csv_path))
        landsat_path = os.path.join(root_path, "landsat_features_training.csv")
        terra_path = os.path.join(root_path, "terraclimate_features_training.csv")

        # (1) 기본 수질 관측 데이터
        self.df = pd.read_csv(csv_path, engine="python")
        self.df.columns = self.df.columns.str.strip()

        # (2) Landsat 피처
        landsat_df = pd.read_csv(landsat_path, engine="python")
        landsat_df.columns = landsat_df.columns.str.strip()

        # (3) TerraClimate 피처
        terra_df = pd.read_csv(terra_path, engine="python")
        terra_df.columns = terra_df.columns.str.strip()

        # (4) 세 데이터프레임을 위치/날짜 기준으로 병합
        merge_keys = ["Latitude", "Longitude", "Sample Date"]
        landsat_cols = merge_keys + ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]
        self.df = pd.merge(self.df, landsat_df[landsat_cols], on=merge_keys, how="left")
        self.df = pd.merge(
            self.df, terra_df[merge_keys + ["pet"]], on=merge_keys, how="left"
        )

        # 탭 피처 컬럼 정의 (총 9개)
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

        # 날짜를 datetime으로 변환 (칩 매칭/보간 시 사용)
        self.df["dt"] = pd.to_datetime(
            self.df["Sample Date"], dayfirst=True, errors="coerce"
        )
        self.df = self.df.dropna(subset=["dt"])

        # 2. 칩 메타데이터 스캔 (위치/날짜 정보를 가진 딕셔너리 리스트)
        self.chips = []
        npy_pattern = os.path.join(chips_dir, "**", "*.npy")

        print(f"🔍 [Loader] Scanning chip files in {chips_dir}...")
        npy_files = glob.glob(npy_pattern, recursive=True)

        for f_path in tqdm(npy_files, desc="📂 [1/2] Parsing Chips", unit="file"):
            parent_dir = os.path.basename(os.path.dirname(f_path))
            f_name = os.path.basename(f_path)

            # 폴더 이름 예시: loc_-28.7608_17.7302
            m_loc = re.search(r"loc_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)", parent_dir)
            # 파일 이름에서 날짜 추출 예시: chip_2011-02-01.npy
            m_date = re.search(r"(\d{4}-\d{2}-\d{2})", f_name)

            if m_loc and m_date:
                lat, lon = float(m_loc.group(1)), float(m_loc.group(2))
                date_str = m_date.group(1)
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    self.chips.append({"lat": lat, "lon": lon, "dt": dt, "path": f_path})
                except ValueError:
                    continue

        # 3. 각 수질 관측(row)에 대해, 가장 적절한 칩(또는 보간)을 찾아 samples 리스트에 저장
        self.samples = []
        target_cols = [
            "Total Alkalinity",
            "Electrical Conductance",
            "Dissolved Reactive Phosphorus",
        ]

        for _, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc="🔗 [2/2] Matching Samples",
            unit="row",
        ):
            # 관측 위치 근처의 칩들만 필터링 (위도/경도 오차 허용)
            pos_chips = [
                c
                for c in self.chips
                if abs(c["lat"] - row["Latitude"]) < 0.001
                and abs(c["lon"] - row["Longitude"]) < 0.001
            ]
            if not pos_chips:
                continue

            tabular_values = row[self.tab_cols].values.astype(np.float32)

            # 3-1. 같은 날짜의 칩이 있으면 그대로 사용 (single)
            exact = [c for c in pos_chips if c["dt"].date() == row["dt"].date()]
            if exact:
                self.samples.append(
                    {
                        "type": "single",
                        "chip": exact[0]["path"],
                        "tabular": tabular_values,
                        "label": row[target_cols].values.astype(np.float32),
                    }
                )
            else:
                # 3-2. 같은 날짜가 없다면, 가장 가까운 과거/미래 칩 두 개로 시간 보간(interpolation)
                pos_chips.sort(key=lambda x: x["dt"])
                prev_c = [c for c in pos_chips if c["dt"] < row["dt"]]
                next_c = [c for c in pos_chips if c["dt"] > row["dt"]]
                if prev_c and next_c:
                    p, n = prev_c[-1], next_c[0]
                    total_diff = (n["dt"] - p["dt"]).total_seconds()
                    alpha = (
                        (row["dt"] - p["dt"]).total_seconds() / total_diff
                        if total_diff > 0
                        else 0
                    )
                    self.samples.append(
                        {
                            "type": "interp",
                            "prev": p["path"],
                            "next": n["path"],
                            "alpha": alpha,
                            "tabular": tabular_values,
                            "label": row[target_cols].values.astype(np.float32),
                        }
                    )

        print(f"✅ [Loader] Ready! Generated {len(self.samples)} training samples.")

    def _load_chip_tensor(self, path: str) -> torch.Tensor:
        """
        .npy 파일을 읽어 PyTorch 텐서(C, H, W) 형태로 변환합니다.
        - 스케일 값에 따라 Landsat/S2 스케일링을 적용하고,
        - 값 범위를 [0, 1]로 클리핑합니다.
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
            # 읽기 실패 시, 0으로 채워진 기본 텐서를 반환
            return torch.zeros((5, 32, 32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        - 반환 이미지: 5채널 위성 칩 텐서 (C, H, W)
        - 반환 탭 피처: 9차원 (위도/경도 + Landsat/TerraClimate 피처)
        - 반환 라벨: 3차원 (3개 수질 지표), log1p 변환된 값
        """
        item = self.samples[idx]

        if item["type"] == "interp":
            # 과거/미래 칩을 alpha 비율로 선형 보간
            img_prev = self._load_chip_tensor(item["prev"])
            img_next = self._load_chip_tensor(item["next"])
            img = (1 - item["alpha"]) * img_prev + item["alpha"] * img_next
        else:
            img = self._load_chip_tensor(item["chip"])

        # 타겟 라벨에 로그 변환 적용 (값의 스케일을 줄여 학습 안정화)
        label_log = np.log1p(item["label"])

        return img, torch.from_numpy(item["tabular"]), torch.from_numpy(label_log)


