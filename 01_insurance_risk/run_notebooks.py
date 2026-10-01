"""Run the notebooks in order, each in a fresh kernel, and record the result.

    python run_notebooks.py            # all notebooks, from a clean state (deletes the freshness manifest)
    python run_notebooks.py 5 7        # only notebooks 05 to 07 (manifest is kept)

Executed outputs are saved back into the notebooks. The run report goes to
results/validation/validation.json. The script stops at the first failing notebook
and exits with a non-zero status.
"""
from pathlib import Path
import importlib.metadata
import json
import os
import re
import sys
import tempfile
import time

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'results' / 'validation' / 'validation.json'
MANIFEST = ROOT / 'results' / 'validation' / 'manifest.json'
PACKAGES = ['numpy', 'scipy', 'pandas', 'matplotlib', 'statsmodels', 'chainladder',
            'scikit-learn', 'nbformat', 'nbclient']


def notebook_paths():
    paths = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    numbers = [int(re.match(r'(\d+)_', p.name).group(1)) for p in paths]
    if numbers != list(range(1, len(paths) + 1)):
        raise SystemExit(f'Notebook numbering must be 01..NN without gaps, found {numbers}')
    return paths


def main(argv):
    paths = notebook_paths()
    first, last = (int(argv[0]), int(argv[1])) if len(argv) == 2 else (1, len(paths))
    full_run = (first, last) == (1, len(paths))

    cache = Path(tempfile.gettempdir()) / 'risk-engine-runtime'
    cache.mkdir(exist_ok=True)
    for var, sub in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython'), ('NUMBA_CACHE_DIR', 'numba')]:
        os.environ.setdefault(var, str(cache / sub))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if full_run and MANIFEST.exists():
        MANIFEST.unlink()   # forces every notebook to regenerate its own inputs
    report = {'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in PACKAGES},
              'notebooks': [], 'status': 'running'}
    REPORT.write_text(json.dumps(report, indent=2))

    for number, path in enumerate(paths, start=1):
        if not first <= number <= last:
            continue
        print(f'RUN  {path.name}', flush=True)
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs, cell.execution_count = [], None
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, timeout=3600, km=manager, resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
            nbformat.validate(notebook)
        except Exception as error:
            nbformat.write(notebook, path)
            report['status'] = 'failed'
            report['notebooks'].append({'name': path.name, 'status': 'failed', 'error': str(error)[-2000:]})
            REPORT.write_text(json.dumps(report, indent=2))
            print(f'FAIL {path.name}: {str(error)[-1500:]}', flush=True)
            raise SystemExit(1)
        finally:
            manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)
        report['notebooks'].append({'name': path.name, 'status': 'passed',
                                    'seconds': round(time.monotonic() - started, 1),
                                    'code_cells': sum(c.cell_type == 'code' for c in notebook.cells)})
        REPORT.write_text(json.dumps(report, indent=2))
        print(f'PASS {path.name} ({time.monotonic() - started:.0f}s)', flush=True)

    report['status'] = 'passed' if full_run else 'partial'
    REPORT.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
