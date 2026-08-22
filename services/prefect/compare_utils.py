import math

def should_promote(new_auc: float, prod_auc: float, delta: float = 0.01) -> bool:
    if prod_auc is None:
        return True
    if isinstance(prod_auc, float) and math.isnan(prod_auc):
        return True
    if new_auc > prod_auc + delta:
        return True
    return False
