import os
import shutil

BACKUP_DIR = "_backup"

def deep_cleanup():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"백업 폴더 생성: {BACKUP_DIR}")

    # os.walk의 topdown=False 옵션으로 가장 깊은 폴더부터 조사
    for root, dirs, files in os.walk('.', topdown=False):
        # 백업 폴더나 숨김 폴더(.git 등)는 건너뜀
        if BACKUP_DIR in root or '/.' in root:
            continue

        for name in dirs:
            dir_path = os.path.join(root, name)

            # 폴더가 비어있는지 확인
            if not os.listdir(dir_path):
                dest_path = os.path.join(BACKUP_DIR, name)

                # 백업 폴더 내 이름 중복 방지
                if os.path.exists(dest_path):
                    dest_path = f"{dest_path}_{int(os.path.getmtime(dir_path))}"

                print(f"[이동] 빈 폴더 발견: {dir_path}")
                shutil.move(dir_path, dest_path)

if __name__ == "__main__":
    deep_cleanup()
