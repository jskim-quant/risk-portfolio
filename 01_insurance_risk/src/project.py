"""Project paths, plot settings and result freshness checks."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
# name -> (sub-directory of results/, notebook that writes it)
RESULT_FILES = {
    'calibrated_model.json':      ('calibration', 1),
    'loss_summary.json':          ('aggregate', 2),
    'stress_results.json':        ('aggregate', 3),
    'reinsurance_results.json':   ('aggregate', 4),
    'multi_line_results.json':    ('portfolio', 5),
    'marginal_robustness.json':   ('portfolio', 5),
    'dependency_results.json':    ('portfolio', 6),
    'portfolio_reinsurance.json': ('portfolio', 6),
    'pricing_results.json':       ('pricing', 7),
    'reserving_results.json':     ('reserving', 8),
    'reserve_risk_summary.json':  ('reserving', 9),
    'scr_results.json':           ('regulatory', 10),
    'integrated_risk.json':       ('summary', 11),
    'parameter_risk.json':        ('summary', 11),
    'model_risk_table.json':      ('summary', 11),
}
PRODUCERS = {name: nb for name, (_, nb) in RESULT_FILES.items()}
VALIDATION_DIR = RESULTS_DIR / 'validation'
MANIFEST_PATH = VALIDATION_DIR / 'manifest.json'


def result_path(name):
    """Return the path for a registered result file."""
    return RESULTS_DIR / RESULT_FILES[name][0] / name


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hash():
    """Hash source files, notebook code and input data."""
    h = hashlib.sha256()
    for path in sorted(DATA_DIR.glob('*.csv')) + sorted((ROOT / 'src').glob('*.py')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        nb = json.loads(path.read_text())
        h.update(path.name.encode())
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                h.update(''.join(cell['source']).encode())
    return h.hexdigest()

def require_results(*names):
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    current = source_hash()
    for name in names:
        path = result_path(name)
        record = manifest.get(name, {})
        if not path.exists() or record.get('source_hash') != current or record.get('sha256') != file_hash(path):
            raise RuntimeError(f'{name} is missing or stale. Run notebooks 01-{PRODUCERS[name]:02d} in order, or python run_notebooks.py.')

def record_results(*names):
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    fingerprint = source_hash()
    for name in names:
        manifest[name] = dict(producer=PRODUCERS[name], sha256=file_hash(result_path(name)),
                              source_hash=fingerprint, generated_utc=datetime.now(timezone.utc).isoformat())
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

def setup():
    import matplotlib.pyplot as plt
    import pandas as pd
    from cycler import cycler
    for subdir in {d for d, _ in RESULT_FILES.values()}:
        (RESULTS_DIR / subdir).mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    font_keys = ('font.family', 'font.size', 'axes.titlesize', 'axes.labelsize',
                 'legend.fontsize', 'xtick.labelsize', 'ytick.labelsize')
    plt.rcParams.update({key: plt.rcParamsDefault[key] for key in font_keys})
    plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 110, 'axes.grid': True,
        'grid.alpha': 0.2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.prop_cycle': cycler(color=['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9'])})
    pd.set_option('display.float_format', lambda value: f'{value:,.3f}')
    pd.set_option('display.max_columns', 12)
