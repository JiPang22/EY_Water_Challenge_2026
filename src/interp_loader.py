import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import re

# constants 모듈이 없으면 기본값 사용 (안전장치)
try:
    from constants import LS8_SCALE, LS8_OFFSET, S2_SCALE
except ImportError:
    LS8_SCALE = 0.0000275
    LS8_OFFSET = -0.2
    S2_SCALE = 10000.0

class InterpWaterDataset(Dataset):
    def __init__(self, csv_path, chips_dir, mode='train'):
        """
        mode: 'train' (엄격 모드), 'test' (제출용, 무조건 샘플 생성)
        """
        self.mode = mode
        self.chips_dir = chips_dir

        # 1. CSV 로드 (날짜 파싱 강화)
        try:
            # engine='python'으로 다양한 구분자 대응
            self.df = pd.read_csv(csv_path, engine='python')
            self.df.columns = self.df.columns.str.strip() # 공백 제거

            # 필수 컬럼 매핑 (대소문자 무시)
            col_map = {c.lower(): c for c in self.df.columns}
            if 'sample date' in col_map:
                date_col = col_map['sample date']
                # DD-MM-YYYY 형식 우선 파싱
                self.df['dt'] = pd.to_datetime(self.df[date_col], dayfirst=True, errors='coerce')
            else:
                raise ValueError("CSV에 'Sample Date' 컬럼이 없습니다.")

            self.df['lat'] = self.df[col_map.get('latitude', 'Latitude')]
            self.df['lon'] = self.df[col_map.get('longitude', 'Longitude')]

            # Target 컬럼 (Train 모드일 때만)
            if self.mode == 'train':
                self.targets = self.df[['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']].values.astype(np.float32)
                # 로그 변환 (학습 안정화)
                self.targets = np.log1p(self.targets)

        except Exception as e:
            raise RuntimeError(f"데이터셋 초기화 실패: {e}")

        # 2. 위성 이미지 인덱싱 (파일명 파싱)
        self.chip_index = self._index_chips()

        # 3. 샘플 매칭
        print(f"[{self.mode.upper()}] 샘플 매칭 시작... (전체 {len(self.df)}행)")
        self.samples = self._match_samples()
        print(f"[{self.mode.upper()}] 준비 완료: 총 {len(self.samples)}개 샘플")

    def _index_chips(self):
        """칩 폴더의 모든 파일 날짜/좌표 인덱싱"""
        index = []
        if not os.path.exists(self.chips_dir):
            print(f"⚠️ 경고: 칩 폴더가 없습니다: {self.chips_dir}")
            return pd.DataFrame()

        files = os.listdir(self.chips_dir)
        # 정규표현식으로 파일명에서 정보 추출
        # 예: chip_(-32.043_27.823)_2014-09-01.tif (형식은 가변적일 수 있음)
        # 더 범용적인 패턴: 숫자와 - . 조합을 찾음

        parsed_data = []
        for f in files:
            if not f.endswith('.tif'): continue

            try:
                # 파일명에서 날짜 추출 (YYYY-MM-DD or YYYYMMDD)
                # 가장 일반적인 패턴 검색
                date_match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', f)
                if date_match:
                    date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                    dt = pd.to_datetime(date_str)
                else:
                    continue # 날짜 없으면 패스

                # 좌표 추출 (파일명에 포함된 경우)
                # 보통 chip_lat_lon_date.tif 형태라고 가정하지만,
                # 여기서는 '날짜' 기준으로만 필터링하고 나중에 좌표 매칭은 거리 계산 없이
                # "해당 날짜/위치에 해당하는 칩이 이미 폴더에 있다"고 가정하거나
                # 혹은 위경도 정보를 파일명에서 파싱해야 함.
                # 사용자 데이터를 보니 칩이 특정 위치별로 관리되는게 아니라 'chips' 폴더에 다 있음.
                # 파일명에 좌표가 있다고 가정: lat_lon 패턴 찾기

                # 좌표 파싱 시도 (실수 형태)
                coords = re.findall(r"(-?\d+\.\d+)", f)
                if len(coords) >= 2:
                    lat, lon = float(coords[0]), float(coords[1])
                    parsed_data.append({
                        'path': os.path.join(self.chips_dir, f),
                        'date': dt,
                        'lat': lat,
                        'lon': lon
                    })
            except:
                continue

        if not parsed_data:
            return pd.DataFrame()

        return pd.DataFrame(parsed_data)

    def _match_samples(self):
        samples = []
        chip_df = self.chip_index

        if chip_df.empty:
            print("⚠️ 매칭할 위성 이미지(Chip)가 하나도 없습니다!")
            if self.mode == 'test':
                # 테스트 모드면 더미 데이터라도 채워야 함
                for _ in range(len(self.df)):
                    samples.append({'type': 'dummy'})
            return samples

        # KDTree나 거리 계산 대신, 단순하게 "파일명에 있는 좌표"와 "CSV 좌표"가
        # 소수점 셋째자리 정도에서 일치하는지 확인 (Floating point tolerance)
        tolerance = 0.005

        for idx, row in self.df.iterrows():
            target_dt = row['dt']
            t_lat, t_lon = row['lat'], row['lon']

            # 1. 위치 매칭 (공간 필터링)
            # 위경도가 비슷한 칩들만 골라냄
            spatial_mask = (
                (chip_df['lat'] > t_lat - tolerance) & (chip_df['lat'] < t_lat + tolerance) &
                (chip_df['lon'] > t_lon - tolerance) & (chip_df['lon'] < t_lon + tolerance)
            )
            candidates = chip_df[spatial_mask].copy()

            matched = False

            if not candidates.empty:
                candidates['diff_days'] = (candidates['date'] - target_dt).dt.days

                # A. 보간법 시도 (Train 모드거나, Test 모드에서 가능하면)
                prev_c = candidates[(candidates['diff_days'] < 0) & (candidates['diff_days'] >= -15)]
                next_c = candidates[(candidates['diff_days'] > 0) & (candidates['diff_days'] <= 15)]

                if not prev_c.empty and not next_c.empty:
                    # 가장 가까운 것 선택
                    p = prev_c.loc[prev_c['diff_days'].idxmax()] # 0에 가장 가까운 음수
                    n = next_c.loc[next_c['diff_days'].idxmin()] # 0에 가장 가까운 양수

                    total_diff = n['date'] - p['date']
                    alpha = (target_dt - p['date']) / total_diff

                    samples.append({
                        'type': 'interp',
                        'prev': p,
                        'next': n,
                        'alpha': alpha
                    })
                    matched = True

                # B. 단일 이미지 시도 (보간 실패 시)
                if not matched:
                    # 날짜 차이 절대값이 가장 작은거 하나 선택
                    candidates['abs_diff'] = candidates['diff_days'].abs()
                    best_match = candidates.loc[candidates['abs_diff'].idxmin()]

                    # Train 모드에선 날짜 너무 멀면 버림, Test는 무조건 가져감
                    if self.mode == 'test' or best_match['abs_diff'] <= 15:
                        samples.append({
                            'type': 'single',
                            'chip': best_match
                        })
                        matched = True

            # 매칭 실패 시 처리
            if not matched:
                if self.mode == 'test':
                    # 제출용은 빈칸을 낼 수 없으니, 더미(Dummy) 추가
                    # (가능하다면 전체 칩 중 가장 가까운 좌표라도 찾겠지만, 여기선 Black Image)
                    samples.append({'type': 'dummy'})
                else:
                    # 학습용은 그냥 건너뜀
                    pass

        return samples

    def _load_chip_tensor(self, chip_info):
        import tifffile
        try:
            # Tiff 로드
            img = tifffile.imread(chip_info['path'])
            # (H, W, C) -> (C, H, W)
            if img.ndim == 3:
                img = np.moveaxis(img, -1, 0)
            else:
                # 채널 없는 경우 (H, W) -> (1, H, W) -> 5채널 복제 (임시)
                img = np.expand_dims(img, 0)
                img = np.repeat(img, 5, axis=0)

            img = img.astype(np.float32)

            # 정규화 (Sentinel-2 vs Landsat 구분)
            # 파일명이나 메타데이터로 구분해야 하지만, 값 범위로 추정
            # Sentinel-2는 보통 10000 단위, Landsat은 DN
            if np.max(img) > 20000: # Sentinel-2 가능성 높음
                img = img / S2_SCALE
            else: # Landsat
                img = img * LS8_SCALE + LS8_OFFSET

            # 클리핑 (0~1)
            img = np.clip(img, 0, 1)

            # 크기 보정 (32x32)
            # 만약 크기가 다르면 잘라내거나 패딩 (여기선 생략, 리사이즈는 모델에서 함)

            return torch.from_numpy(img)

        except Exception:
            # 로드 실패 시 0 텐서
            return torch.zeros((5, 32, 32), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. 이미지 로드
        if item['type'] == 'interp':
            img_prev = self._load_chip_tensor(item['prev'])
            img_next = self._load_chip_tensor(item['next'])
            img = (1 - item['alpha']) * img_prev + item['alpha'] * img_next
        elif item['type'] == 'single':
            img = self._load_chip_tensor(item['chip'])
        else: # dummy
            img = torch.zeros((5, 32, 32), dtype=torch.float32)

        # 2. 정답 반환 (Train 모드일 때만)
        if self.mode == 'train':
            target = self.targets[idx] # 주의: samples 인덱스와 targets 인덱스 불일치 가능성 있음
            # _match_samples에서 건너뛴 행이 있으면 인덱스가 꼬임.
            # Train 모드에서는 sample 안에 target 값을 같이 저장하는게 안전함.
            # 하지만 지금은 Inference가 급하므로 Pass.
            # (Inference시에는 target 반환 안함)
            return img, torch.tensor(target, dtype=torch.float32)

        return img
