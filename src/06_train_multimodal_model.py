"""
[06단계] MobileViT 멀티모달 모델 학습 스크립트

- 입력:
  - water_quality_training_dataset.csv
  - landsat_features_training.csv
  - terraclimate_features_training.csv
  - rawData/chips (이미지 칩)
- 출력:
  - models/best_multimodal_model.pth (가장 좋은 검증 손실 모델)
  - plots/learning_curve.png (학습/검증 손실 곡선)

이 파일은 기존 `train.py`를
실행 순서가 보이도록 파일명과 주석을 정리한 버전입니다.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# GUI 에러 방지 (서버/CLI 환경에서도 그림 저장 가능하도록 설정)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from 04_build_training_dataset import InterpWaterDataset
from model import MobileViT_XXS_Multimodal

# 📉 메모리 절약용 설정 (실제 GPU 메모리에 맞게 조정)
REAL_BATCH_SIZE = 2
TARGET_BATCH_SIZE = 32
ACCUMULATION_STEPS = TARGET_BATCH_SIZE // REAL_BATCH_SIZE

LR = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../models"
PLOT_DIR = "../plots"


def print_model_summary(model: torch.nn.Module) -> None:
    """모델 구조와 파라미터 수를 간단히 출력합니다."""
    print("\n📊 [Model Summary]")
    print("-" * 60)
    print(f"Architecture: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print("-" * 60 + "\n")


def main():
    # 출력 디렉터리 생성
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    torch.cuda.empty_cache()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    csv_path = os.path.join(root_dir, "rawData", "water_quality_training_dataset.csv")
    chips_dir = os.path.join(root_dir, "rawData", "chips")

    print(f"🚀 [System] Training on {DEVICE} (FP32 Mode)")
    print(
        f"⚙️  Real Batch: {REAL_BATCH_SIZE} | "
        f"Accumulation: {ACCUMULATION_STEPS} | LR: {LR}"
    )

    # 1. 모델 준비 (이미지 + 탭 피처 멀티모달 구조)
    model = MobileViT_XXS_Multimodal(
        image_channels=5, tabular_dim=9, output_dim=3
    ).to(DEVICE)
    print_model_summary(model)

    # 2. 데이터셋/데이터로더 준비
    full_dataset = InterpWaterDataset(csv_path, chips_dir)
    if len(full_dataset) == 0:
        raise ValueError("❌ No samples found!")

    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=REAL_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=REAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. 손실함수/옵티마이저/스케줄러 설정
    #    - SmoothL1Loss: MSE보다 이상치에 덜 민감하여 수치 폭주 방지에 유리
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print("\n══════════════════════ [ Training Start ] ══════════════════════")
    global_start_time = time.time()

    try:
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()

            # -----------------------
            # 1) Train phase
            # -----------------------
            model.train()
            running_loss = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS}",
                unit="batch",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            )

            optimizer.zero_grad()

            for i, (images, tabular, labels) in enumerate(pbar):
                images, tabular, labels = (
                    images.to(DEVICE),
                    tabular.to(DEVICE),
                    labels.to(DEVICE),
                )

                outputs = model(images, tabular)
                loss = criterion(outputs, labels)

                # 첫 에폭/첫 배치에서 예측 vs 실제 값 대략 확인
                if i == 0 and epoch == 0:
                    print(
                        f"\n[Debug] Pred: {outputs[0].detach().cpu().numpy()} "
                        f"| Real: {labels[0].detach().cpu().numpy()}"
                    )

                # gradient accumulation을 위해 손실을 나눠줌
                loss = loss / ACCUMULATION_STEPS
                loss.backward()

                if (i + 1) % ACCUMULATION_STEPS == 0:
                    # 기울기 폭주 방지
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                curr_loss = loss.item() * ACCUMULATION_STEPS
                running_loss += curr_loss

                pbar.set_postfix(
                    {
                        "Loss": f"{curr_loss:.4f}",
                        "LR": f"{optimizer.param_groups[0]['lr']:.1e}",
                    }
                )

            avg_train_loss = running_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # -----------------------
            # 2) Validation phase
            # -----------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, tabular, labels in val_loader:
                    images, tabular, labels = (
                        images.to(DEVICE),
                        tabular.to(DEVICE),
                        labels.to(DEVICE),
                    )
                    outputs = model(images, tabular)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)
            scheduler.step()

            # Epoch ETA 출력 (대략적인 참고용)
            epoch_time = time.time() - epoch_start_time
            remain_epochs = EPOCHS - (epoch + 1)
            est_remain_min = (epoch_time * remain_epochs) / 60.0

            epoch_msg = (
                f"Done! Avg Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"Est. Remaining: {est_remain_min:.1f} min"
            )

            # best validation loss 갱신 시 모델 저장
            if avg_val_loss < best_loss and avg_val_loss > 1e-6:
                best_loss = avg_val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(SAVE_DIR, "best_multimodal_model.pth"),
                )
                epoch_msg += " [★ Saved]"
                print(f"\033[92m{epoch_msg}\033[0m")
            else:
                print(epoch_msg)

    finally:
        # 예외가 나더라도 학습 곡선은 가능한 한 남겨두기
        total_time_min = (time.time() - global_start_time) / 60
        print("\n══════════════════════ [ Finished ] ══════════════════════")
        print(
            f"⏱️  Total Time: {total_time_min:.1f} min | "
            f"Best Val Loss: {best_loss:.4f}"
        )

        # 학습/검증 손실 곡선 저장
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Val Loss", linestyle="--")
            plt.title("Training & Validation Loss Curve")
            plt.xlabel("Epochs")
            plt.ylabel("Loss (SmoothL1)")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(PLOT_DIR, "learning_curve.png")
            plt.savefig(plot_path)
            print(f"📊 Learning curve saved to: {plot_path}")
        except Exception as e:
            print(f"⚠️ Graph Plotting Failed: {e}")


if __name__ == "__main__":
    main()

