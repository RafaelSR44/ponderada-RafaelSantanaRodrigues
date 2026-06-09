import numpy as np


def cross_entropy(y_pred_probs, y_true_onehot, eps=1e-12):
    """
    y_pred_probs: (batch, classes) softmax output
    y_true_onehot: (batch, classes) one-hot encoded labels
    """
    y_pred_probs = np.clip(y_pred_probs, eps, 1.0 - eps)
    n = y_pred_probs.shape[0]
    return -np.sum(y_true_onehot * np.log(y_pred_probs)) / n


def cross_entropy_softmax_gradient(y_pred_probs, y_true_onehot):
    """
    Combined gradient of cross-entropy loss w.r.t. softmax input (pre-activation z).
    Derivation: dL/dz = (softmax(z) - y) / batch_size
    """
    n = y_pred_probs.shape[0]
    return (y_pred_probs - y_true_onehot) / n
