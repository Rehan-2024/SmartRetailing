"""Execute the delivered notebook locally, preserving genuine cell outputs."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
path=root/'notebooks/SmartRetail_All_Models_Colab.ipynb'
notebook=nbformat.read(path,as_version=4)
def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'Running cell {cell_index+1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:85]}',flush=True)
client=NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(root)}},allow_errors=False,on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook,path)
nbformat.validate(notebook)
errors=[out for cell in notebook.cells if cell.cell_type=='code' for out in cell.outputs if out.output_type=='error']
if errors:raise RuntimeError('Notebook contains execution errors')
print(f'Execution finished: {path.stat().st_size:,} bytes; no cell errors.',flush=True)
