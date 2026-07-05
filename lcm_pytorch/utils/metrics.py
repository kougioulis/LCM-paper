from typing import Union
import numpy as np
import torch
import torchmetrics


def custom_binary_metrics(
        pred: Union[torch.Tensor, np.ndarray],
        A: Union[torch.Tensor, np.ndarray],
        verbose: bool=True, 
        threshold: float=0.05
):
    """ 
    Computes the AUC, TPR, FPR, TNR and FNR on the inferred lagged adjacency tensor.
    Adjusted from https://github.com/Gideon-Stein/CausalPretraining/tree/main.

    Args:
        pred (torch.Tensor or np.ndarray) : The predicted temporal adjacency matrix (should NOT be thresholded) of shape `(n, n, l_max)`
        A (torch.Tensor or np.ndarray) : The ground truth temporal adjacency matrix of shape `(n, n, l_max)` 
        verbose (bool) : Whether to print or not the results (verbose logging)
        threshold (float) : The threshold used for binary convertion, before the calculation of assistive binary metrics. Default is `0.05`. 

    Returns (tuple): The TPR, FPR, TNR, FNR and AUC scores
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)

    pred = pred.cpu()
    A = A.cpu()

    A_bin = (A >= 0.05).int()
    pred_bin = (pred >= threshold).int()

    # Compute AUC before thresholding
    auc_metric = torchmetrics.classification.BinaryAUROC()
    auc = auc_metric(pred.flatten(), A_bin.flatten()).item()

    tp = torch.sum((pred_bin == 1) & (A_bin == 1)).item()
    tn = torch.sum((pred_bin == 0) & (A_bin == 0)).item()
    fp = torch.sum((pred_bin == 1) & (A_bin == 0)).item()
    fn = torch.sum((pred_bin == 0) & (A_bin == 1)).item()

    denom_pos = tp + fn
    denom_neg = fp + tn

    tpr = tp / denom_pos if denom_pos > 0 else 0.0
    fpr = fp / denom_neg if denom_neg > 0 else 0.0
    tnr = tn / denom_neg if denom_neg > 0 else 0.0
    fnr = fn / denom_pos if denom_pos > 0 else 0.0

    if verbose:
        print(f"Ground truth edges: {A_bin.sum().item()}")
        print(f"Predicted edges: {pred_bin.sum().item()}")
        print(f"AUC: {auc:.4f}")
        print(f"TPR: {tpr:.4f}, FPR: {fpr:.4f}, TNR: {tnr:.4f}, FNR: {fnr:.4f}")
        print()
    
    return tpr, fpr, tnr, fnr, auc
    


def calc_roc_curve(binary: Union[torch.Tensor, np.ndarray], A: Union[torch.Tensor, np.ndarray]):
    """Calculates ROC and AUROC scores. 
    Args:
        binary (numpy.array or torch.Tensor): The predicted lagged adjacency tensor of shape `(n, n, l_max)`
        A (numpy.array or torch.Tensor): The ground truth lagged adjacency tensor of shape `(n, n, l_max)`

    Returns:
        roc_out: The Receiver Operating Characteristic (ROC) curve
        auroc_out: The Area Under the Receiver Operating Characteristic (AUROC) curve
    """
    if isinstance(out, np.ndarray):
        out = torch.tensor(np.ascontiguousarray(out), dtype=torch.float32) # guarantee contiguous memory allocation
        
    if isinstance(A, np.ndarray):
        A = torch.tensor(np.ascontiguousarray(A), dtype=torch.float32)

    roc = torchmetrics.classification.BinaryROC()
    auroc = torchmetrics.classification.BinaryAUROC()
    roc_out = roc(preds=torch.Tensor(binary), target=A.type(torch.int64))
    auroc_out = auroc(preds=torch.Tensor(binary), target=A.type(torch.int64))

    return roc_out, auroc_out


def SHD(target: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray], double_for_anticausal: bool=True, threshold: float=0.05, normalize: bool=False):
    """
    Compute the Structural Hamming Distance (SHD) for lagged adjacency tensors (instead of static prediction tensors).

    Args:
        target (np.ndarray or torch.Tensor of shape `(num_vars, num_vars, max_lag)`):
                Ground truth binary adjacency tensor.
        pred (np.ndarray or torch.Tensor of same shape):
              Predicted adjacency tensor.
        double_for_anticausal (bool): If `True`, counts reversed edges as two mistakes. Default is `True`.
        threshold (float): Threshold to binarize predicted adjacencies. Default is `0.05`.
        normalize (bool): If `True`, normalize the SHD score by the maximum possible SHD score. Default is `False`.

    Returns (int): SHD score across all lags.

    Notes:
        Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The Max-min Hill-Climbing Bayesian Network Structure Learning Algorithm.
        Machine learning, 65, 31-78.
    """
    if hasattr(target, 'detach'): # tensor to np
        target = target.detach().cpu().numpy()
    if hasattr(pred, 'detach'):
        pred = pred.detach().cpu().numpy()

    V, _, L = target.shape
    shd_tot = 0

    for lag in range(L):
        true_adj = (target[:, :, lag] > 0).astype(int)
        pred_adj = (pred[:, :, lag] >= threshold).astype(int)

        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                if true_adj[i, j] == pred_adj[i, j]:
                    continue
                # extra or missing edge
                if true_adj[j, i] == pred_adj[j, i]:
                    shd_tot += 1
                # reversed edge
                else:
                    shd_tot += 2 if double_for_anticausal else 1

    if normalize:
        possible_edges = V * (V - 1) * L
        shd_norm = shd_tot / possible_edges
        return shd_tot, shd_norm
    else:
        return shd_tot