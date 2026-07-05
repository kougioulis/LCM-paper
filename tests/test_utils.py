"""
Tests for the tensor helpers.
"""

import torch

from src.utils.metrics import custom_binary_metrics
from src.utils.utils import corr_regularization, lagged_batch_crosscorrelation


def test_binary_metrics_perfect_prediction():
    """
    Perfect predictions give TPR=1, FPR=0, TNR=1, FNR=0 (returns floats).
    """
    A = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pred = torch.tensor([0.9, 0.0, 0.8, 0.01])  # above/below the 0.05 threshold
    tpr, fpr, tnr, fnr, auc = custom_binary_metrics(pred, A, verbose=False)

    assert tpr == 1.0
    assert fpr == 0.0
    assert tnr == 1.0
    assert fnr == 0.0
    assert 0.0 <= auc <= 1.0


def test_lagged_batch_crosscorrelation_shape():
    B, N, D, max_lags = 3, 40, 4, 3
    corr = lagged_batch_crosscorrelation(torch.randn(B, N, D), max_lags)

    assert corr.shape == (B, D, D * max_lags)
    assert torch.isfinite(corr).all()


def test_corr_regularization_is_nonnegative_scalar():
    B, D, max_lag = 2, 4, 3
    data = torch.randn(B, 40, D)
    predictions = torch.rand(B, D, D, max_lag)  # in [0, 1] like sigmoid output
    loss = corr_regularization(predictions, data)

    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)
