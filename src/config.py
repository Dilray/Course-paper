import os
from pathlib import Path

SEED = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

STOP_WORDS = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'is', 'are', 'it', 'this', 'that']
MAX_FEATURES = 5000