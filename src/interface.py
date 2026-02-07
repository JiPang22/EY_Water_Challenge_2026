from abc import ABC, abstractmethod
import time

class WaterQualityDatasetInterface(ABC):
    @abstractmethod
    def __len__(self): pass
    @abstractmethod
    def __getitem__(self, idx): pass

class WaterQualityModelInterface(ABC):
    @abstractmethod
    def forward(self, satellite_data, climate_data): pass

def print_eta(start_time, current, total):
    elapsed = time.time() - start_time
    if current < 0: return
    avg = elapsed / (current + 1)
    remaining = total - (current + 1)
    eta = avg * remaining
    print(f"Progress: {current+1}/{total} | ETA: {eta:.2f}s")
