import os
import pandas as pd
import pystac_client
import planetary_computer
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
)

DATA_DIR = "/mnt/data_lake/EY_Satellite_Raw"
MAX_WORKERS = 16
BANDS = ["B02", "B03", "B04", "B08"]
console = Console(force_terminal=True)

def download_asset(item, asset_key, save_dir, prefix, progress, file_task, byte_task):
    if asset_key not in item.assets:
        progress.update(file_task, advance=1)
        return False
    
    url = item.assets[asset_key].href
    timestamp = item.datetime.strftime("%Y%m%d")
    filename = f"{prefix}_{timestamp}_{asset_key}.tif"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        # 이미 있는 파일은 개수만 올리고 종료
        progress.update(file_task, advance=1)
        return True

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            # 파일 실제 크기를 가져와서 byte_task에 추가
            file_size = int(r.headers.get('content-length', 0))
            progress.update(byte_task, total=progress.tasks[byte_task].total + file_size)
            
            console.print(f"[bold cyan]GET : {filename[:30]}... ({file_size/1024/1024:.1f}MB)[/bold cyan]")
            
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(byte_task, advance=len(chunk))
        
        progress.update(file_task, advance=1)
        return True
    except Exception as e:
        with open("download_errors.log", "a") as f:
            f.write(f"{filename}: {str(e)}\n")
        progress.update(file_task, advance=1)
        return False

def run_collector():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    df = pd.read_csv('data/raw/train.csv')
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    download_tasks = []
    console.print("[bold cyan]▶ SEARCHING...[/bold cyan]")
    
    for _, row in df.iterrows():
        bbox = [row['longitude'] - 0.01, row['latitude'] - 0.01, row['longitude'] + 0.01, row['latitude'] + 0.01]
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=row['date'], query={"eo:cloud_cover": {"lt": 15}})
        items = list(search.items())
        for item in items:
            for band in BANDS:
                download_tasks.append((item, band, DATA_DIR, row['id']))

    console.print(f"[bold green]✔ FOUND {len(download_tasks)} FILES[/bold green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=5
    ) as progress:
        # 두 개의 바를 운영: 파일 개수용, 데이터 용량용
        file_task = progress.add_task("Files", total=len(download_tasks))
        byte_task = progress.add_task("Data ", total=0) # 다운로드 시작할 때 total이 늘어남

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(download_asset, *t, progress, file_task, byte_task) for t in download_tasks]
            for _ in as_completed(futures):
                pass

if __name__ == "__main__":
    run_collector()
