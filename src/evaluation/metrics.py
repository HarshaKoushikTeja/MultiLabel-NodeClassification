from sklearn.metrics import f1_score, hamming_loss as hl
import numpy as np


def micro_f1(y_true, y_pred):
    """Compute Micro F1 Score"""
    return f1_score(y_true, y_pred, average='micro', zero_division=0)


def macro_f1(y_true, y_pred):
    """Compute Macro F1 Score"""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_hamming_loss(y_true, y_pred):
    """Compute Hamming Loss"""
    return hl(y_true, y_pred)
