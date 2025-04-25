
"""utils.paths – centralised project paths"""
from pathlib import Path

# climb ONE level: utils/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_paths():
    data_proc = _PROJECT_ROOT / 'data' / 'processed'
    return {
        'root': _PROJECT_ROOT,
        'data': data_proc,
        'merged': data_proc / 'merged_gw_cleaned.csv',
        'teams':  data_proc / 'teams.csv',
        'fixtures': data_proc / 'fixtures.csv',
        'models': _PROJECT_ROOT / 'models'
    }
