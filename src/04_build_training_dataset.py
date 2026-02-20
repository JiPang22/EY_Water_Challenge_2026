"""
[04단계] PyTorch Dataset 정의

- 역할:
  1) CSV(학습 데이터 + Landsat + Terra)를 로드하고 병합합니다.
  2) 저장된 StandardScaler를 불러와 탭 피처를 정규화합니다.
  3) 날짜 정보를 이용해 Month Sin/Cos 피처를 추가합니다 (총 9개 탭 피처).
  4) 위도/경도/날짜에 맞는 이미지 칩(.npy)을 로드합니다.
  5) 타겟 값에 log1p 변환을 적용하여 반환합니다.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class InterpWaterDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        chips_dir: str,
        scaler_path: str = "models/scaler_tabular.pkl",
        mode: str = "train",
    ):
        """
        Args:
            csv_path: water_quality_training_dataset.csv 경로
            chips_dir: 이미지 칩이 저장된 디렉터리 (rawData/chips)
            scaler_path: 학습된 StandardScaler pkl 경로
            mode: 'train' or 'test' (타겟 유무에 따라 다름)
        """
        self.chips_dir = chips_dir
        self.mode = mode

        # 1. 경로 설정 (csv_path 기준 root 추론)
        if "rawData" in csv_path:
            root_dir = csv_path.split("rawData")[0]
        else:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))

        # 2. 데이터 로드
        train_df = pd.read_csv(csv_path)
        
        # 학습용/테스트용 피처 파일명 선택
        if mode == "train":
            landsat_name = "landsat_features_training.csv"
            terra_name = "terraclimate_features_training.csv"
        else:
            landsat_name = "landsat_features_test.csv"
            terra_name = "terraclimate_features_test.csv"

        landsat_path = os.path.join(root_dir, landsat_name)
        terra_path = os.path.join(root_dir, terra_name)

        # 3. 병합
        self.df = self._merge_data(train_df, landsat_path, terra_path)

        # 4. 스케일러 로드
        if not os.path.isabs(scaler_path):
            scaler_path = os.path.join(root_dir, scaler_path)
        
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            print(f"⚠️ Warning: Scaler not found at {scaler_path}. Tabular features will not be scaled.")
            self.scaler = None

        # 5. 컬럼 정의
        self.feature_cols = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI", "pet"]
        self.target_cols = [
            "Total Alkalinity",
            "Electrical Conductance",
            "Dissolved Reactive Phosphorus",
        ]

    def _merge_data(self, main_df, landsat_path, terra_path):
        """메인 데이터프레임과 위성/기후 피처를 병합합니다."""
        main_df.columns = main_df.columns.str.strip()
        main_df = self._format_keys(main_df)

        if os.path.exists(landsat_path):
            ldf = pd.read_csv(landsat_path)
            ldf.columns = ldf.columns.str.strip()
            ldf = self._format_keys(ldf)
            cols = ["Latitude", "Longitude", "Sample Date", "nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]
            cols = [c for c in cols if c in ldf.columns]
            main_df = pd.merge(main_df, ldf[cols], on=["Latitude", "Longitude", "Sample Date"], how="left")
        
        if os.path.exists(terra_path):
            tdf = pd.read_csv(terra_path)
            tdf.columns = tdf.columns.str.strip()
            tdf = self._format_keys(tdf)
            cols = ["Latitude", "Longitude", "Sample Date", "pet"]
            cols = [c for c in cols if c in tdf.columns]
            main_df = pd.merge(main_df, tdf[cols], on=["Latitude", "Longitude", "Sample Date"], how="left")
            
        return main_df

    def _format_keys(self, df):
        """병합 키 포맷 통일"""
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").round(6).astype(str)
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").round(6).astype(str)
        df["Sample Date"] = pd.to_datetime(df["Sample Date"], dayfirst=True, errors="coerce").dt.strftime("%d-%m-%Y")
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # [1] Tabular Features (7 scaled + 2 date = 9 dim)
        raw_feats = [row.get(c, 0.0) if not pd.isna(row.get(c, 0.0)) else 0.0 for c in self.feature_cols]
        raw_feats = np.array(raw_feats, dtype=np.float32).reshape(1, -1)
        
        if self.scaler:
            try:
                scaled_feats = self.scaler.transform(raw_feats).flatten()
            except:
                scaled_feats = raw_feats.flatten()
        else:
            scaled_feats = raw_feats.flatten()

        try:
            dt = datetime.strptime(str(row["Sample Date"]), "%d-%m-%Y")
            month = dt.month
        except:
            month = 1
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        tabular = np.concatenate([scaled_feats, [month_sin, month_cos]]).astype(np.float32)

        # [2] Image Chip (5 channels)
        try:
            lat_f = float(row["Latitude"])
            lon_f = float(row["Longitude"])
            loc_id = f"{lat_f:.4f}_{lon_f:.4f}"
            date_str = datetime.strptime(str(row["Sample Date"]), "%d-%m-%Y").strftime("%Y-%m-%d")
            
            chip_path = os.path.join(self.chips_dir, f"loc_{loc_id}", f"{date_str}.npy")
            
            if os.path.exists(chip_path):
                img = np.load(chip_path).astype(np.float32)
                if img.max() > 10.0: img = img / 10000.0
                img = np.clip(img, 0, 1)
            else:
                img = np.zeros((5, 32, 32), dtype=np.float32)
        except:
            img = np.zeros((5, 32, 32), dtype=np.float32)

        # [3] Target (Log1p)
        if self.mode == "train":
            targets = [row.get(c, 0.0) if not pd.isna(row.get(c, 0.0)) else 0.0 for c in self.target_cols]
            targets = np.log1p(np.array(targets, dtype=np.float32))
            return torch.tensor(img), torch.tensor(tabular), torch.tensor(targets)
        else:
            return torch.tensor(img), torch.tensor(tabular)