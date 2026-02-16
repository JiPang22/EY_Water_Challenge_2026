# src/constants.py
# 위성별 정규화 계수 및 수질 타겟 정의

TARGET_COLS = [
    'Total Alkalinity', 
    'Electrical Conductance', 
    'Dissolved Reactive Phosphorus'
]

# Landsat-8 Scaling: DN * 0.0000275 - 0.2
LS8_SCALE = 0.0000275
LS8_OFFSET = -0.2

# Sentinel-2 Scaling: DN / 10000
S2_SCALE = 10000.0

# 최대 허용 날짜 차이 (보간 범위)
MAX_DIFF_DAYS = 15
