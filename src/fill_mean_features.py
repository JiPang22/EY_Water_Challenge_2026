import pandas as pd
import os
import numpy as np

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# 1. 학습용 피처 파일 찾기 (평균 계산용)
train_landsat_path = os.path.join(ROOT_DIR, "rawData", "landsat_features_training.csv")
train_terra_path = os.path.join(ROOT_DIR, "rawData", "terraclimate_features_training.csv")

# 파일이 없으면 루트에서도 검색
if not os.path.exists(train_landsat_path):
    train_landsat_path = os.path.join(ROOT_DIR, "landsat_features_training.csv")
if not os.path.exists(train_terra_path):
    train_terra_path = os.path.join(ROOT_DIR, "terraclimate_features_training.csv")

# 2. 템플릿 파일 로드 (행 개수 확인용)
template_path = os.path.join(ROOT_DIR, "submission_template.csv")
if not os.path.exists(template_path):
    print("❌ submission_template.csv not found!")
    exit()

df_template = pd.read_csv(template_path)
n_rows = len(df_template)
print(f"📄 Template has {n_rows} rows.")

# 3. Landsat 평균 채우기
if os.path.exists(train_landsat_path):
    print("calc mean from Landsat training...")
    df_train = pd.read_csv(train_landsat_path)
    # 필요한 컬럼만 선택
    cols = ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']
    means = df_train[cols].mean()
    
    # 테스트용 DF 생성
    df_test_landsat = pd.DataFrame()
    # Key 컬럼 복사
    df_test_landsat['Latitude'] = df_template['Latitude']
    df_test_landsat['Longitude'] = df_template['Longitude']
    df_test_landsat['Sample Date'] = df_template['Sample Date']
    
    # 평균값으로 채우기
    for c in cols:
        df_test_landsat[c] = means[c]
        
    save_path = os.path.join(ROOT_DIR, "landsat_features_test.csv")
    df_test_landsat.to_csv(save_path, index=False)
    print(f"✅ Created {save_path} (filled with means)")

# 4. TerraClimate 평균 채우기
if os.path.exists(train_terra_path):
    print("calc mean from TerraClimate training...")
    df_train = pd.read_csv(train_terra_path)
    cols = ['pet'] # TerraClimate 주요 피처
    means = df_train[cols].mean()
    
    df_test_terra = pd.DataFrame()
    df_test_terra['Latitude'] = df_template['Latitude']
    df_test_terra['Longitude'] = df_template['Longitude']
    df_test_terra['Sample Date'] = df_template['Sample Date']
    
    for c in cols:
        df_test_terra[c] = means[c]
        
    save_path = os.path.join(ROOT_DIR, "terraclimate_features_test.csv")
    df_test_terra.to_csv(save_path, index=False)
    print(f"✅ Created {save_path} (filled with means)")

print("\n🚀 Now run 'python3 src/inference.py' again!")
