import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime
import re
import glob

# 상수 모듈 로드 시도
try:
    from constants import LS8_SCALE, LS8_OFFSET, S2_SCALE
except ImportError:
    LS8_SCALE = 0.0000275
    LS8_OFFSET = -0.2
    S2_SCALE = 10000.0

class InterpWaterDataset(Dataset):
    def __init__(self, csv_path, chips_dir, mode='train'):
        self.mode = mode
        self.chips_dir = chips_dir

        # 1. CSV 경로 설정
        root_path = os.path.dirname(os.path.dirname(csv_path))
        landsat_path = os.path.join(root_path, 'landsat_features_training.csv')
        terra_path = os.path.join(root_path, 'terraclimate_features_training.csv')

        # 2. 데이터 프레임 로드
        self.df = pd.read_csv(csv_path, engine='python')
        self.df.columns = self.df.columns.str.strip()

        landsat_df = pd.read_csv(landsat_path, engine='python')
        landsat_df.columns = landsat_df.columns.str.strip()

        terra_df = pd.read_csv(terra_path, engine='python')
        terra_df.columns = terra_df.columns.str.strip()

        # 3. 피처 병합
        merge_keys = ['Latitude', 'Longitude', 'Sample Date']
        landsat_cols = merge_keys + ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']
        self.df = pd.merge(self.df, landsat_df[landsat_cols], on=merge_keys, how='left')
        self.df = pd.merge(self.df, terra_df[merge_keys + ['pet']], on=merge_keys, how='left')

        self.tab_cols = ['Latitude', 'Longitude', 'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI', 'pet']
        self.df[self.tab_cols] = self.df[self.tab_cols].fillna(0)

        self.df['dt'] = pd.to_datetime(self.df['Sample Date'], dayfirst=True, errors='coerce')
        self.df = self.df.dropna(subset=['dt'])

        # 4. 이미지 칩(.npy) 재귀적 탐색
        self.chips = []
        npy_pattern = os.path.join(chips_dir, "**", "*.npy")
        # glob으로 모든 .npy 파일 수집
        npy_files = glob.glob(npy_pattern, recursive=True)

        print(f"[Loader] Found {len(npy_files)} .npy files in {chips_dir}")

        for f_path in npy_files:
            # 경로 구조: .../loc_-22.22_29.99/2014-09-13.npy
            # 상위 폴더명에서 lat, lon 추출
            parent_dir = os.path.basename(os.path.dirname(f_path))
            f_name = os.path.basename(f_path)

            m_loc = re.search(r"loc_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)", parent_dir)
            m_date = re.search(r"(\d{4}-\d{2}-\d{2})", f_name)

            if m_loc and m_date:
                lat, lon = float(m_loc.group(1)), float(m_loc.group(2))
                date_str = m_date.group(1)
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                    self.chips.append({'lat': lat, 'lon': lon, 'dt': dt, 'path': f_path})
                except ValueError:
                    continue

        print(f"[Loader] Successfully parsed {len(self.chips)} chips.")

        # 5. 샘플 매칭
        self.samples = []
        target_cols = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

        for _, row in self.df.iterrows():
            pos_chips = [c for c in self.chips if abs(c['lat']-row['Latitude']) < 0.001 and abs(c['lon']-row['Longitude']) < 0.001]
            if not pos_chips: continue

            tabular_values = row[self.tab_cols].values.astype(np.float32)
            exact = [c for c in pos_chips if c['dt'].date() == row['dt'].date()]

            if exact:
                self.samples.append({
                    'type': 'single',
                    'chip': exact[0]['path'],
                    'tabular': tabular_values,
                    'label': row[target_cols].values.astype(np.float32)
                })
            else:
                pos_chips.sort(key=lambda x: x['dt'])
                prev_c = [c for c in pos_chips if c['dt'] < row['dt']]
                next_c = [c for c in pos_chips if c['dt'] > row['dt']]
                if prev_c and next_c:
                    p, n = prev_c[-1], next_c[0]
                    total_diff = (n['dt'] - p['dt']).total_seconds()
                    alpha = (row['dt'] - p['dt']).total_seconds() / total_diff if total_diff > 0 else 0
                    self.samples.append({
                        'type': 'interp',
                        'prev': p['path'],
                        'next': n['path'],
                        'alpha': alpha,
                        'tabular': tabular_values,
                        'label': row[target_cols].values.astype(np.float32)
                    })

        print(f"[Loader] Generated {len(self.samples)} training samples.")

    def _load_chip_tensor(self, path):
        try:
            # .npy 로드 (numpy array)
            img = np.load(path).astype(np.float32)

            # (H, W, C) -> (C, H, W) 변환
            # 보통 .npy는 (32, 32, 5) 형태로 저장되는 경우가 많음
            if img.ndim == 3 and img.shape[-1] == 5:
                img = img.transpose(2, 0, 1)
            elif img.ndim == 2: # 채널이 1개인 경우 복제
                img = np.stack([img]*5, axis=0)

            # 스케일링 로직
            if np.max(img) > 20000:
                img = img / S2_SCALE
            elif np.max(img) > 1: # Landsat DN값일 경우
                img = img * LS8_SCALE + LS8_OFFSET

            return torch.from_numpy(np.clip(img, 0, 1))
        except Exception:
            return torch.zeros((5, 32, 32), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if item['type'] == 'interp':
            img_prev = self._load_chip_tensor(item['prev'])
            img_next = self._load_chip_tensor(item['next'])
            img = (1 - item['alpha']) * img_prev + item['alpha'] * img_next
        else:
            img = self._load_chip_tensor(item['chip'])
        return img, torch.from_numpy(item['tabular']), torch.from_numpy(item['label'])
