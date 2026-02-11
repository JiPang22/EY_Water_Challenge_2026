import os
import shutil

BACKUP_DIR = "_backup"

def move_empty_or_nested_folders():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    # 현재 디렉토리의 모든 항목 조사
    for entry in os.listdir('.'):
        if entry == BACKUP_DIR or entry.startswith('.'):
            continue

        if os.path.isdir(entry):
            # 폴더 내부가 비어있거나, 중첩된 빈 폴더만 있는 경우
            full_path = os.path.abspath(entry)
            is_empty = not any(os.scandir(full_path))

            if is_empty:
                print(f"[이동] 빈 폴더: {entry} -> {BACKUP_DIR}")
                shutil.move(entry, os.path.join(BACKUP_DIR, entry))
            else:
                # 특정 패턴(예: tif 추출 후 남은 찌꺼기 폴더)이 있다면 추가 로직 가능
                pass

if __name__ == "__main__":
    move_empty_or_nested_folders()
