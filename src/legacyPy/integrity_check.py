import os
import glob
import numpy as np
from tqdm import tqdm

def check_integrity():
    files = glob.glob('rawData/chips/loc_*/*.npy')
    corrupted = []
    
    print(f'🔍 총 {len(files)}개 파일 무결성 검사 시작...')
    
    for f in tqdm(files):
        try:
            data = np.load(f)
            # 차원(Dimension) 및 데이터 존재 여부 확인
            if data.shape != (5, 32, 32):
                corrupted.append(f)
        except Exception:
            corrupted.append(f)
            
    if corrupted:
        print(f'✘ {len(corrupted)}개의 손상된 파일 발견. 삭제를 시작합니다.')
        for f in corrupted:
            os.remove(f)
            # 쌍이 되는 메타데이터 파일도 함께 삭제
            meta_f = f.replace('.npy', '_meta.json')
            if os.path.exists(meta_f):
                os.remove(meta_f)
        print('✔ 삭제 완료. 수집기를 다시 실행하여 빈자리를 채우세요.')
    else:
        print('✔ 모든 파일이 정상입니다. 찐빠 없음.')

if __name__ == "__main__":
    check_integrity()
