import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 커스텀 모듈 임포트
from interp_loader import InterpWaterDataset
from model import MobileViT_XXS
from constants import TARGET_COLS

# --------------------------------------------------------
# 1. 하이퍼파라미터 설정
# --------------------------------------------------------
BATCH_SIZE = 32
LR = 1e-3        # 0.001 (초기 학습률)
EPOCHS = 50      # 총 학습 횟수
VAL_SPLIT = 0.2  # 검증 데이터 비율 (20%)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "../models"

def main():
    # 모델 저장 폴더 생성
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 2. 데이터셋 준비
    # --------------------------------------------------------
    print(f"[Init] 데이터셋 준비 중...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    csv_path = os.path.join(root_dir, 'rawData', 'water_quality_training_dataset.csv')
    chips_dir = os.path.join(root_dir, 'rawData', 'chips')

    # 전체 데이터 로드
    full_dataset = InterpWaterDataset(csv_path, chips_dir)

    # Train / Validation 분할
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 데이터 로더 (num_workers는 CPU 코어 수에 따라 조절, 에러 시 0으로 변경)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[Init] 데이터 분할 완료: Train({train_size}) / Val({val_size})")

    # --------------------------------------------------------
    # 3. 모델 및 학습 도구 설정
    # --------------------------------------------------------
    # img_size=256 설정 (모델 내부 리사이즈와 일치)
    model = MobileViT_XXS(input_channels=5, output_dim=3, img_size=256).to(DEVICE)

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.MSELoss() # 회귀 문제 (평균 제곱 오차)

    # Learning Rate Scheduler (학습률을 서서히 줄임)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --------------------------------------------------------
    # 4. 학습 루프 (Training Loop)
    # --------------------------------------------------------
    best_loss = float('inf')
    start_time = time.time()

    print(f"\n[Train] 학습 시작 (Device: {DEVICE}, Epochs: {EPOCHS})")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # --- Train Mode ---
        model.train()
        train_loss = 0.0

        # 진행률 표시 (tqdm)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()      # 기울기 초기화
            outputs = model(imgs)      # 예측
            loss = criterion(outputs, targets) # 오차 계산
            loss.backward()            # 역전파
            optimizer.step()           # 가중치 갱신

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Mode ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # 스케줄러 업데이트
        scheduler.step()

        # --- 결과 기록 및 저장 ---
        elapsed = time.time() - start_time
        avg_time_per_epoch = elapsed / (epoch + 1)
        remaining_time = avg_time_per_epoch * (EPOCHS - epoch - 1)

        log_msg = (f"Epoch {epoch+1:02d} | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"ETA: {remaining_time/60:.1f}m")

        # Best Model 저장
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            log_msg += " [Saved ★]"

        print(log_msg)

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"[Done] 학습 완료. 총 소요 시간: {total_time/60:.1f}분")
    print(f"[Result] 최종 Best Val Loss: {best_loss:.4f}")
    print(f"[Path] 모델 저장 위치: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    # 실행 전 GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 단위 테스트: 간단한 Forward 실행
    try:
        print("[UnitTest] 사전 점검 중...")
        test_model = MobileViT_XXS(5, 3, 256).to(DEVICE)
        test_in = torch.randn(2, 5, 32, 32).to(DEVICE)
        _ = test_model(test_in)
        print("[UnitTest] 통과. 학습을 시작합니다.")
        main()
    except Exception as e:
        print(f"[Error] 유닛 테스트 실패: {e}")
