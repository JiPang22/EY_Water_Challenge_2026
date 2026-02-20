import nbformat
import os

def convert_and_patch(input_file, output_file, is_landsat=True):
    if not os.path.exists(input_file):
        print(f"❌ {input_file}을 찾을 수 없습니다.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = [cell.source for cell in nb.cells if cell.cell_type == 'code']
    script = "\n\n".join(code_cells)
    
    # 📝 [Path Patching]
    # 입력: 상위 폴더의 submission_template.csv 참조
    script = script.replace("'data/water_quality_validation_dataset.csv'", "'../submission_template.csv'")
    script = script.replace('"data/water_quality_validation_dataset.csv"', '"../submission_template.csv"')
    
    # 출력: 상위 폴더에 test용 피처 파일로 저장
    if is_landsat:
        # 변수명과 저장 경로를 테스트용으로 강제 치환
        script = script.replace("val_features_path = 'landsat_features_validation.csv'", "val_features_path = '../landsat_features_test.csv'")
        script = script.replace("landsat_val_features.to_csv", "print('Saving to ../landsat_features_test.csv...'); landsat_val_features.to_csv")
    else:
        script = script.replace("out_df.to_csv('terraclimate_features_validation.csv', index=False)", "out_df.to_csv('../terraclimate_features_test.csv', index=False)")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Generated and Patched Extractor for Inference\n\n")
        f.write(script)
    print(f"✅ {output_file} 생성 및 경로 패치 완료")

if __name__ == "__main__":
    convert_and_patch('Landsat_Data_Extraction_Notebook.ipynb', 'Landsat_Extractor.py', True)
    convert_and_patch('TerraClimate_Data_Extraction_Notebook.ipynb', 'TerraClimate_Extractor.py', False)
