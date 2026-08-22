import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "services" / "prefect"))
from compare_utils import should_promote

def test_should_promote():
    assert should_promote(new_auc=0.80, prod_auc=0.78, delta=0.01) is True

def test_should_not_promote():
    assert should_promote(new_auc=0.785, prod_auc=0.78, delta=0.01) is False