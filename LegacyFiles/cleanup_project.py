import os
import shutil
import glob
import time
from datetime import timedelta

# 정리 대상 패턴 목록 (필요시 수정)
PATTERNS = [
    "__pycache__",          # 캐시 폴더
    "*.log",                # 로그 파일
    "ventoy*",              # 설치 관련 불필요 파일
    "woeusb*",              # 설치 관련 불필요 파일
    "test_*.py",            # 임시 테스트 스크립트 (tests 폴더 제외)
    "test_*.tif",           # 테스트용 이미지
    "test.wav",             # 테스트용 오디오
    "debug_*.py",           # 디버깅용 코드
    "check_*.py",           # 확인용 코드
    "preview*",             # 프리뷰 파일
    "*_v[0-9]*.py",         # 버전 넘버링된 구버전 파일 (예: patch_generator_v3.py)
    "download_*.py",        # 다운로드 스크립트 (메인 제외 정리)
    "*.tar.gz",             # 압축 파일
    "*.docx",               # 문서 파일
    "*.html",               # html 파일
]

# 제외할 중요 파일 (패턴에 걸려도 이동 안 함)
EXCLUDE_FILES = [
    "download_chips_final.py",
    "train_final.py",
    "main.py"
]

BACKUP_DIR = "_backup"

def cleanup():
    start_time = time.time()
    
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"폴더 생성: {BACKUP_DIR}")

    moved_count = 0
    
    # 패턴별 파일 수집
    files_to_move = set()
    for pattern in PATTERNS:
        found = glob.glob(pattern)
        files_to_move.update(found)

    files_to_move = list(files_to_move)
    total_files = len(files_to_move)

    print(f"정리 대상 파일: {total_files}개")

    for i, file_path in enumerate(files_to_move):
        # 제외 파일 확인
        if file_path in EXCLUDE_FILES:
            continue
            
        # 디렉토리인 경우와 파일인 경우 구분하여 이동
        try:
            dest = os.path.join(BACKUP_DIR, file_path)
            if os.path.exists(dest):
                # 백업 폴더에 이미 있으면 덮어쓰거나 이름 변경 (여기선 건너뜀)
                print(f"[Skip] 이미 백업됨: {file_path}")
                continue
                
            shutil.move(file_path, dest)
            moved_count += 1
            print(f"[이동] {file_path} -> {BACKUP_DIR}/")
        except Exception as e:
            print(f"[Error] {file_path} 이동 실패: {e}")

        # 종료 예상 시간 출력 (단순 파일 이동이라 매우 짧음)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remain = total_files - (i + 1)
            eta = timedelta(seconds=int(remain * avg_time))
            print(f"예상 종료 시간: {eta}")

    print(f"\n총 {moved_count}개 파일이 {BACKUP_DIR} 폴더로 이동되었습니다.")

if __name__ == "__main__":
    cleanup()
