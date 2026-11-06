"""Portable Decision2Care working directories; credentials come from the environment."""
import os
from pathlib import Path
ROOT = Path(os.environ.get('DECISION2CARE_WORKDIR', 'private')).expanduser().resolve()
DATA_DIR = ROOT / 'data'
CDM_DIR = Path(os.environ.get('DECISION2CARE_RESOURCES', str(DATA_DIR / 'resources'))).expanduser().resolve()
CACHE_DIR = ROOT / 'cache'
OUTPUTS_DIR = ROOT / 'outputs'
def load_typesafe_key():
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if not key:
        raise RuntimeError('Set TYPESAFE_API_KEY before live inference.')
    return key
