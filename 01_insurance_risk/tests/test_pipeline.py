"""Pipeline hygiene and invariants on the saved results (skipped when results have not been generated)."""
import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import project
from conftest import ROOT

RESULTS = ROOT / 'results'


def load(name):
    path = project.result_path(name)
    if not path.exists():
        pytest.skip(f'{name} not generated; run python run_notebooks.py')
    return json.loads(path.read_text())


def test_artifact_freshness_tracks_content_and_source():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for folder in ('data', 'src', 'notebooks', 'results'):
            (root / folder).mkdir()
        source = root / 'src/model.py'; source.write_text('version = 1')
        artifact = root / 'results/calibration/calibrated_model.json'
        with patch.multiple(project, ROOT=root, DATA_DIR=root / 'data', RESULTS_DIR=root / 'results',
                            VALIDATION_DIR=root / 'results/validation', MANIFEST_PATH=root / 'results/validation/manifest.json'):
            project.setup()
            artifact.write_text('{}')
            with pytest.raises(RuntimeError):
                project.require_results('calibrated_model.json')
            project.record_results('calibrated_model.json')
            project.require_results('calibrated_model.json')
            artifact.write_text('{"changed": true}')
            with pytest.raises(RuntimeError):
                project.require_results('calibrated_model.json')
            project.record_results('calibrated_model.json')
            source.write_text('version = 2')
            with pytest.raises(RuntimeError):
                project.require_results('calibrated_model.json')


def test_result_registry_is_complete():
    assert set(project.RESULT_FILES) == set(project.PRODUCERS)
    assert sorted(set(project.PRODUCERS.values())) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_no_absolute_local_paths_in_code_or_notebooks():
    pattern = re.compile(r'(/Users/|/home/\w+|[A-Za-z]:\\\\Users|/mnt/)')
    files = list((ROOT / 'src').glob('*.py')) + [ROOT / 'run_notebooks.py'] + list((ROOT / 'tests').glob('*.py'))
    for nb in (ROOT / 'notebooks').glob('*.ipynb'):
        cells = json.loads(nb.read_text())['cells']
        text = '\n'.join(''.join(c['source']) for c in cells)
        assert not pattern.search(text), nb.name
    for path in files:
        if path.name == 'test_pipeline.py':
            continue
        assert not pattern.search(path.read_text()), path.name


def test_no_generated_files_committed_to_repo_tree():
    ignored = (ROOT / '.gitignore').read_text()
    for entry in ('__pycache__/', '.ipynb_checkpoints/', '.DS_Store'):
        assert entry in ignored
    assert not (ROOT / 'test_models.py').exists() and not (ROOT / 'validation.json').exists()


def test_units_and_labels_in_saved_results():
    assert load('loss_summary.json')['currency'] == 'EUR'
    assert load('pricing_results.json')['currency'] == 'EUR'
    assert load('multi_line_results.json')['unit'] == 'USD thousands'
    for name in ('dependency_results.json', 'portfolio_reinsurance.json', 'integrated_risk.json', 'marginal_robustness.json'):
        assert load(name)['unit'] == 'USD millions'
    assert load('reserving_results.json')['currency'] == 'USD thousands'
    scr = load('scr_results.json')
    assert scr['currency'] == 'EUR million' and 'not a total SCR' in scr['scope']


def test_loss_summary_invariants():
    s = load('loss_summary.json')
    assert s['es_995'] >= s['var_995'] > 0
    assert s['capital_proxy_995'] == pytest.approx(s['var_995'] - s['analytical_mean_loss'], rel=1e-9)
    assert s['es_995_replicates_500k']['p05'] <= s['es_995_replicates_500k']['p95']


def test_portfolio_invariants():
    m = load('multi_line_results.json')
    assert m['portfolio_var995'] < m['sum_standalone_var995']
    assert 0 < m['d_var'] < m['d_k'] < 1
    d = load('dependency_results.json')
    sc = {r['scenario']: r for r in d['copula_scenarios']}
    assert sc['Comonotonic (perfect positive dependence)']['D_VaR'] == pytest.approx(0, abs=1e-3)   # sample-quantile interpolation
    assert all(r['ES995'] >= r['VaR995'] for r in d['copula_scenarios'])
    var_stress = [r['VaR995'] for r in d['correlation_stress']]
    assert var_stress == sorted(var_stress)
    assert sum(r['Allocated ES995 share (Euler)'] for r in d['capital_attribution']) == pytest.approx(1, abs=1e-9)
    assert d['latent_checks']['gaussian']['min_eigenvalue'] > 0


def test_portfolio_reinsurance_invariants():
    rows = load('portfolio_reinsurance.json')['results']
    for dep in {r['dependence'] for r in rows}:
        gross = next(r for r in rows if r['dependence'] == dep and r['treaty'] == 'Gross')
        for r in (x for x in rows if x['dependence'] == dep and x['treaty'] != 'Gross'):
            assert r['VaR995'] < gross['VaR995'] and r['ES995'] < gross['ES995']
            assert r['VaR_relief'] == pytest.approx(gross['VaR995'] - r['VaR995'], abs=1e-9)
            assert r['expected_ceded'] > 0


def test_integrated_risk_and_waterfall_invariants():
    r = load('integrated_risk.json')
    sc = r['scenarios']
    assert [s['VaR_total'] for s in sc] == sorted(s['VaR_total'] for s in sc)
    assert all(s['K_total'] >= s['K_premium'] - 1e-9 for s in sc)
    steps = r['waterfall']['steps']
    running = steps[0]['value_usd_m']
    for step in steps[1:]:
        if step['kind'] == 'delta':
            running += step['value_usd_m']
        else:
            assert step['value_usd_m'] == pytest.approx(running, rel=1e-9)
    for row in r['net_of_reinsurance']:
        assert row['VaR_relief'] >= -1e-9 and row['ES'] > 0


def test_parameter_risk_and_model_risk_table():
    p = load('parameter_risk.json')
    for case in ('cas', 'motor'):
        for row in p[case]['summary'] + p[case]['es_summary']:
            assert row['p05'] <= row['median'] <= row['p95']
    cas = {r['source']: r for r in p['cas']['summary']}
    assert cas['Monte Carlo noise only']['sd'] < cas['Parameters: Marginals and dependence']['sd']
    table = load('model_risk_table.json')['rows']
    assert len(table) >= 10 and {'CAS USD m', 'Motor EUR'} <= {r['basis'] for r in table}
    assert all(r['impact'] == pytest.approx(r['alternative_VaR995'] - r['base_VaR995']) for r in table)


def test_reserve_and_pricing_results():
    per_line = load('reserve_risk_summary.json')['per_line']
    assert {r['line'] for r in per_line} == {'ppauto', 'comauto', 'wkcomp', 'medmal', 'othliab', 'prodliab'}
    assert all(r['std_error'] > 0 and r['reserve_to_premium'] > 0 for r in per_line)
    sens = load('pricing_results.json')['capital_loading']['sensitivity']
    assert [s['k'] for s in sens] == [0.0, 0.005, 0.01, 0.015, 0.02]
    assert [s['avg_gross_premium_eur'] for s in sens] == sorted(s['avg_gross_premium_eur'] for s in sens)
