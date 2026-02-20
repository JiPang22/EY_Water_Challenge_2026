import nbformat
import sys

def convert_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = [cell.source for cell in nb.cells if cell.cell_type == 'code']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Generated from " + input_file + "\n\n")
        f.write("\n\n".join(code_cells))
    print(f"✅ Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    # Landsat 변환
    convert_notebook('rawData/Landsat_Data_Extraction_Notebook.ipynb', 'Landsat_Extractor.py')
    # TerraClimate 변환
    convert_notebook('rawData/TerraClimate_Data_Extraction_Notebook.ipynb', 'TerraClimate_Extractor.py')
