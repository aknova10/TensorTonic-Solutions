import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.array(fpr, dtype=np.float32)
    tpr = np.array(tpr, dtype=np.float32)
    
    AUC = np.sum((tpr[:-1] + tpr[1:]) * (fpr[1:] - fpr[:-1])) / 2

    return AUC