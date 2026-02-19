"""
[09단계] 서브미션 CSV 생성 전용 스크립트

- 입력:
  - 테스트용 CSV (예: submission_template.csv)
  - 08단계에서 생성된 예측 결과가 들어 있는 submission.csv (또는 별도 예측 파일)
- 출력:
  - 컬럼 순서와 형식을 최종 점검한 submission_final.csv

현재 파이프라인에서는 08단계에서 이미 submission.csv를 완성해 저장합니다.
이 스크립트는 필요 시:
- 컬럼 순서를 다시 한 번 확인하거나
- 후처리(라운딩, 특정 범위 클리핑 등)를 추가하는 용도로 사용할 수 있습니다.
"""

import os

import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    # 기본 파일 경로
    template_path = os.path.join(root_dir, "submission_template.csv")
    submission_path = os.path.join(root_dir, "submission.csv")
    output_path = os.path.join(root_dir, "submission_final.csv")

    if not os.path.exists(submission_path):
        print("❌ Base submission.csv not found. Run 08_run_inference_on_test.py first.")
        return

    # 1. 템플릿/서브미션 로드
    if os.path.exists(template_path):
        template_df = pd.read_csv(template_path)
        template_df.columns = template_df.columns.str.strip()
        template_cols = template_df.columns.tolist()
    else:
        template_df = None
        template_cols = None

    sub_df = pd.read_csv(submission_path)
    sub_df.columns = sub_df.columns.str.strip()

    # 2. 필요 시 컬럼 순서를 템플릿에 맞춤
    if template_cols is not None:
        # 템플릿에 없는 타겟 컬럼이 있다면 뒤에 추가
        extra_cols = [c for c in sub_df.columns if c not in template_cols]
        final_cols = template_cols + [c for c in extra_cols if c not in template_cols]
        # 빠진 컬럼은 있으면 NaN으로 채워 넣음
        for c in final_cols:
            if c not in sub_df.columns:
                sub_df[c] = None
        sub_df = sub_df[final_cols]

    # 3. (선택) 후처리: 음수 클리핑, 소수점 자리수 조정 등
    target_cols = [
        "Total Alkalinity",
        "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]
    for col in target_cols:
        if col in sub_df.columns:
            sub_df[col] = sub_df[col].clip(lower=0.0)

    # 4. 최종 저장
    sub_df.to_csv(output_path, index=False)
    print(f"🎯 Final submission saved to: {output_path}")


if __name__ == "__main__":
    main()

