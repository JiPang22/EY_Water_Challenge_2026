# check_patch.py
import numpy as np
import os

patch_path = "/mnt/data_lake/EY_Satellite_Patches"
sample_file = os.listdir(patch_path)[0]
data = np.load(os.path.join(patch_path, sample_file))

print(f"File: {sample_file}")
print(f"Shape: {data.shape}")
print(f"Max Value: {data.max()}")
print(f"Min Value: {data.min()}")
