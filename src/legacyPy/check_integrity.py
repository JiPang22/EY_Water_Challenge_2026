import os
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def verify_dataset():
    # 1. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    csv_path = os.path.join(project_root, 'rawData', 'water_quality_training_dataset.csv')

    console.print(Panel.fit("🔍 EY Water Challenge 2026 - Integrity Check", style="bold yellow"))

    try:
        if not os.path.exists(csv_path):
            console.print(f"[bold red]✘ 파일 없음:[/bold red] {csv_path}")
            return

        # 2. 데이터 로드 (확인된 컬럼명 반영)
        df = pd.read_csv(csv_path)

        # 3. 날짜 및 위치 유효성 체크 (대문자 및 띄어쓰기 반영)
        # Sample Date, Latitude, Longitude 컬럼 사용
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')

        # 대회 요구 기간: 2011 ~ 2015
        date_mask = df['Sample Date'].dt.year.between(2011, 2015)
        # 남아공 좌표 범위: Lat(-35~-22), Lon(16~33)
        loc_mask = df['Latitude'].between(-35, -22) & df['Longitude'].between(16, 33)

        # 전체 정상 데이터 필터
        valid_mask = date_mask & loc_mask

        # 4. 결과 리포트 테이블
        table = Table(title="Data Validation Report", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Status")

        table.add_row("Total Rows", f"{len(df)}", "📋")
        table.add_row("Valid Scope (2011-2015 & SA)", f"{valid_mask.sum()}", "[green]Pass[/green]")
        table.add_row("Out of Scope", f"{len(df) - valid_mask.sum()}", "[yellow]Check Required[/yellow]")
        table.add_row("Date Format Errors", f"{df['Sample Date'].isna().sum()}", "[red]Invalid[/red]")

        console.print(table)

        # 5. 범위 외 데이터가 있을 경우 샘플 출력
        if not valid_mask.all():
            console.print("\n[bold yellow]⚠ 범위 외 데이터 샘플 (최대 3건):[/bold yellow]")
            invalid_samples = df[~valid_mask].head(3)
            console.print(invalid_samples[['Latitude', 'Longitude', 'Sample Date']])

    except Exception as e:
        console.print(f"[bold red]✘ 검증 중 오류 발생:[/bold red] {str(e)}")

if __name__ == "__main__":
    verify_dataset()
