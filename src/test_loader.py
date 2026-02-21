# test_loader.py (Vim)
import torch
from interp_loader import InterpWaterDataset

def test_data_flow():
    ds = InterpWaterDataset(csv_path='...', chips_dir='...')
    img, tab, label = ds[0]
    print(f"Image Dim: {img.shape}") # (5, 32, 32) expected
    print(f"Tabular Dim: {tab.shape}") # (9,) expected
    print(f"Label Dim: {label.shape}") # (3,) expected

if __name__ == "__main__":
    test_data_flow()
