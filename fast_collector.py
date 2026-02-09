import pystac_client
import planetary_computer
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import time

# --- 설정 ---
MAX_WORKERS = 32  # 병렬 프로세스 수
DOWNLOAD_DIR = "data/raw/satellite_test"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- 세션 최적화 (TCP 재사용) ---
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
session.mount('https://', adapter)

print(">>> [1/3] MPC 서버 접속...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# 테스트용 검색 (기간 확대)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[-77.01, 38.89, -76.99, 38.91],
    datetime="2023-01-01/2023-03-30",
    query={"eo:cloud_cover": {"lt": 10}}
)

items = list(search.item_collection())
print(f">>> [2/3] 검색 완료: {len(items)}개 Scene 발견")

# 다운로드 목록 생성 (상위 5개 Scene * 4개 밴드 = 20개 파일)
target_assets = ["B02", "B03", "B04", "B08"]
tasks = []

for item in items[:5]:
    for asset in target_assets:
        if asset in item.assets:
            tasks.append((item.assets[asset].href, os.path.join(DOWNLOAD_DIR, f"{item.id}_{asset}.tif")))

print(f">>> [3/3] 병렬 다운로드 시작 (Workers: {MAX_WORKERS})...")

def download(task):
    url, path = task
    try:
        if os.path.exists(path): return 0
        resp = session.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=32768):
                f.write(chunk)
        return total
    except:
        return 0

# 실행 및 ETA 출력
start = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    results = list(tqdm(pool.map(download, tasks), total=len(tasks), unit="file"))

mb = sum(results) / (1024*1024)
sec = time.time() - start
print(f"\n📊 결과: {mb:.2f} MB / {sec:.2f}초 ({mb/sec:.2f} MB/s)")
