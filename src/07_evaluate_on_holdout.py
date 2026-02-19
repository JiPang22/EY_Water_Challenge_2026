"""
[07단계] 홀드아웃 검증 스크립트

- 입력:
  - water_quality_training_dataset.csv (+ 외부 피처, 칩)
  - models/best_multimodal_model.pth
- 출력:
  - 홀드아웃 세트에 대한 RMSE/MAE/R^2 등 지표
  - 예측 vs 실제 산점도/잔차 플롯 (plots/ 아래 저장)

train 스크립트와 비슷한 방식으로 데이터를 나누되,
검증용 Subset에 대해 모델을 로드하여 성능을 정성적으로 확인하는 용도입니다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from 04_build_training_dataset import InterpWaterDataset
from model import MobileViT_XXS_Multimodal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.2


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    csv_path = os.path.join(root_dir, "rawData", "water_quality_training_dataset.csv")
    chips_dir = os.path.join(root_dir, "rawData", "chips")
    model_path = os.path.join(root_dir, "models", "best_multimodal_model.pth")
    plot_dir = os.path.join(root_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. 데이터셋/홀드아웃 분리
    full_dataset = InterpWaterDataset(csv_path, chips_dir)
    if len(full_dataset) == 0:
        raise ValueError("❌ No samples found!")

    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_ds = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # 2. 모델 로드
    model = MobileViT_XXS_Multimodal(
        image_channels=5, tabular_dim=9, output_dim=3
    ).to(DEVICE)
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. 홀드아웃 예측 수집 (log1p 스케일에서 원 스케일로 복구)
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for images, tabular, labels_log in tqdm(val_loader, desc="Evaluating"):
            images, tabular, labels_log = (
                images.to(DEVICE),
                tabular.to(DEVICE),
                labels_log.to(DEVICE),
            )
            outputs_log = model(images, tabular)

            # log1p 스케일 → 실제 값으로 복구
            preds = torch.expm1(outputs_log).cpu().numpy()
            trues = torch.expm1(labels_log).cpu().numpy()

            all_preds.append(preds)
            all_trues.append(trues)

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    target_names = [
        "Total Alkalinity",
        "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]

    # 4. 지표 계산 및 출력
    print("\n📊 Holdout metrics (per target):")
    for i, name in enumerate(target_names):
        y_true = all_trues[:, i]
        y_pred = all_preds[:, i]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"  - {name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R^2={r2:.3f}")

        # 산점도 플롯
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, alpha=0.4)
        lims = [
            min(np.min(y_true), np.min(y_pred)),
            max(np.max(y_true), np.max(y_pred)),
        ]
        plt.plot(lims, lims, "r--", label="Ideal")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Pred vs True - {name}")
        plt.legend()
        path = os.path.join(
            plot_dir, f"scatter_holdout_{name.replace(' ', '_')}.png"
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    print(f"\n📁 Holdout evaluation plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()

