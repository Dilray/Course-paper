import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.append(str(root))

def test_import():
    from src.data_preprocessing import TextPreprocessor
    assert TextPreprocessor is not None