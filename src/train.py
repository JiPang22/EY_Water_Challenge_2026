import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt  # 시각화 라이브러리 추가
from torch.cuda.amp import GradScaler, autocast # 혼합 정밀도(AMP) 추가

# 1. 커스텀 모듈 로드
from interp_loader import InterpWaterDataset
from model import MobileViT_XXS_Multimodal

# 2. 하이퍼파라미터 (메모리 최적화 적용)
BATCH_SIZE = 8   # 📉 32 -> 8 (OOM 해결 핵심)
LR = 1e-3
EPOCHS = 50
VAL_SPLIT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "../models"
PLOT_DIR = "../plots" # 그래프 저장 경로

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    csv_path = os.path.join(root_dir, 'rawData', 'water_quality_training_dataset.csv')
    chips_dir = os.path.join(root_dir, 'rawData', 'chips')

    print(f"🚀 [System] Initializing Training on {DEVICE} (Batch: {BATCH_SIZE})")

    # 데이터셋 로드
    full_dataset = InterpWaterDataset(csv_path, chips_dir)
    if len(full_dataset) == 0:
        raise ValueError("❌ No samples found!")

    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ [Data] Loaded: Train {len(train_ds)} | Val {len(val_ds)}")

    model = MobileViT_XXS_Multimodal(image_channels=5, tabular_dim=9, output_dim=3).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ⚡ AMP Scaler 초기화 (메모리 절약 및 속도 향상)
    scaler = GradScaler()

    best_loss = float('inf')

    # 📊 학습 기록 저장용 리스트
    history = {'train_loss': [], 'val_loss': []}

    print("\n══════════════════════ [ Training Start ] ══════════════════════")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")

        for images, tabular, labels in pbar:
            images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            # ⚡ Mixed Precision Forward
            with autocast():
                outputs = model(images, tabular)
                loss = criterion(outputs, labels)

            # ⚡ Mixed Precision Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            curr_loss = loss.item()
            running_loss += curr_loss
            pbar.set_postfix({"Loss": f"{curr_loss:.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.1e}"})

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images, tabular)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        scheduler.step()

        epoch_msg = f"Done! Avg Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_multimodal_model.pth"))
            epoch_msg += " [★ Saved]"
            print(f"\033[92m{epoch_msg}\033[0m")
        else:
            print(epoch_msg)

    total_time = (time.time() - start_time) / 60
    print("\n══════════════════════ [ Finished ] ══════════════════════")
    print(f"⏱️  Total Time: {total_time:.1f} min | Best Val Loss: {best_loss:.4f}")

    # 📈 학습 곡선 그리기 및 저장
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss (MSE)')
    plt.plot(history['val_loss'], label='Val Loss (MSE)', linestyle='--')
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(PLOT_DIR, "learning_curve.png")
    plt.savefig(plot_path)
    print(f"📊 Learning curve saved to: {plot_path}")

    # (선택) GUI 환경이면 창 띄우기 시도
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
