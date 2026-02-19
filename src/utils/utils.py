import os
import string  # for labels in the graph
import sys
import warnings
from functools import wraps
from pathlib import Path
from time import time

import einops
import numpy as np
import pandas as pd
import torch
import torchmetrics
import yaml
from omegaconf import OmegaConf
from statsmodels.tsa.stattools import adfuller
from tigramite import data_processing as pp
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr_wls import ParCorr
from tigramite.pcmci import PCMCI
from lingam import VARLiNGAM

from torchmetrics.classification import BinaryAUROC

sys.path.append("..")

def load_config(config_path: Path):
    """
    Loads a config file and returns it as a dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return OmegaConf.to_container(config, resolve=True)

def get_device() -> str:
    """
    Returns the device available to torch.
    
    Args:
    ---
       None
    
    Returns:
    ---
       - `cuda` if CUDA is currently available, else `cpu`.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _from_cp_to_full(adj_cp: torch.Tensor, node_names: str=None) -> pd.DataFrame:

    """
    From CP-style lagged adjacency matrix to full-time-graph.
    Made as a separate method to avoid boilerplate code.

    Args
    ----
        - adj_cp (np.array) : CP-style lagged adjacency matrix, as a Numpy array of shape (n_vars, n_vars, n_lags)
        - node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; 
                        if None, it follows an alphabetical order
    
    Return
    ------ 
        temp_adj_pd: the full-time-graph adjacency matrix as a pd.DataFrame
    """
    n_vars = adj_cp.shape[1]
    n_lags = adj_cp.shape[2]

    # Get default current node names if not provided
    if not node_names:
        node_names = (list(string.ascii_uppercase) + list(string.ascii_lowercase))[:n_vars]
    t_node_names = [x + "_t" for x in node_names]

    # Get lagged node names and create the corresponding DataFrames
    lagged_adj_pd_list = []
    lagged_node_names_list = [t_node_names]
    for t in range(n_lags):
        # create the names
        lagged_node_names = [x + f"-{n_lags-t}" for x in t_node_names]
        lagged_node_names_list.append(lagged_node_names)
        # create the dataframe
        lagged_adj_pd = pd.DataFrame(data=adj_cp[:, :, t].T, columns=t_node_names, index=lagged_node_names) 
        lagged_adj_pd_list.append(lagged_adj_pd)

    # Create the unrolled adjacency DataFrame
    sorted_names = list(sorted([y for x in lagged_node_names_list for y in x], key=lambda f: f.split("_t-")[-1] if "_t-" in f else "0"))
    temp_adj_pd = pd.DataFrame(
        data=np.zeros(shape=(len(sorted_names), len(sorted_names))), 
        columns=sorted_names, #columns=reversed(sorted_names), 
        index=sorted_names, #index=reversed(sorted_names), 
        dtype=int
    )
    for lagged_adj_pd in lagged_adj_pd_list:
        for row in lagged_adj_pd.index:
            for col in lagged_adj_pd.columns:
                temp_adj_pd.loc[row, col] = lagged_adj_pd.loc[row, col]

    return temp_adj_pd

# Timing decorator for functions
def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time()
        result = func(*args, **kwargs)
        tac = time()
        print('Elapsed time: %2.4f sec' % (np.round(tac-tic, 4)))
        return result, np.round(tac - tic, 4)

    return wrapper

    
def binary_metrics(pred: torch.Tensor, A: torch.Tensor, verbose: bool=False, threshold: float=0.05):
    """ 
    Computation of binary metrics on the inferred lagged adjacency tensor.
    Adjusted from https://github.com/Gideon-Stein/CausalPretraining/tree/main.
    
    Args
    -----
        - pred (torch.tensor): The predicted `(n_vars x n_vars x max_lag)` temporal adjacency matrix (should NOT be thresholded)
        - A (torch.tensor): The ground truth `(n_vars x n_vars x max_lag)` temporal adjacency matrix  
        - verbose (bool): Whether to print or not the results (default: `True`)

    Returns (list)
    ------
        - metrics (list): A list of computed metrics (TPR, FPR, TNR, FNR, AUC)
    """
    A = A.clone()
    pred = pred.clone()

    # Convert ground truth to binary - might not always be required
    A = (A >= threshold).float()
    
    # Compute AUC
    auc_metric = torchmetrics.classification.BinaryAUROC().to(pred.device)
    auc = auc_metric(pred, A)
    
    # Convert predictions to binary - might not always be required
    binary = (pred >= threshold).float()

    # true positive - false positive - true negative - false negative
    tp = torch.sum((binary == 1) * (A == 1))
    tn = torch.sum((binary == 0) * (A == 0))
    fp = torch.sum((binary == 1) * (A == 0))
    fn = torch.sum((binary == 0) * (A == 1))

    # true positive % - false positive % - true negative % - false negative %
    tpr, fpr, tnr, fnr = tp / (tp + fn), fp / (fp + tn), tn / (fp + tn), fn / (tp + fn)

    return tpr, fpr, tnr, fnr, auc


def transform_corr_to_y(corr: torch.tensor, max_lag: int, n_vars: int) -> torch.tensor:
    """
    Transforms lagged crosscorrelation batch input of shape [batch_size, num_vars, num_vars, max_lag] to a flattened crosscorrelation input 
    """ 
    corr = einops.rearrange(corr, 'b c1 (t c2) -> b c1 c2 t', c2=n_vars, t=max_lag)

    return torch.flip(corr, dims=[3]) # reverse lag dimension so that lag=0 is most recent


def lagged_batch_crosscorrelation(points: torch.tensor, max_lags: int, eps: float=1e-6):
    """
    Computes lagged cross-correlations between variables at different time lags,
    concatenating lagged variables along the batch dimension. It normalizes the result
    using standard deviations to produce correlation coefficients.
    
    Args:
        points (torch.Tensor): Input tensor of shape (B, N, D) where:
            - B: batch size
            - N: time dimension (number of time steps)
            - D: number of variables/features
        max_lags (int): Maximum number of lags to consider for the cross-correlation.
        eps (float, optional): Small epsilon value for numerical stability to prevent
            division by zero and NaN/Inf values. Defaults to 1e-6.
    
    Returns:
        torch.Tensor: Lagged cross-correlation matrix of shape (B, D, D) containing
            only forward-in-time correlations (excluding backwards-in-time links).
            Each element [i, j] represents the correlation between variable i at the
            current time and variable j at different lags.
    
    Notes:
        - Uses an unbiased covariance estimate (division by N-1).
        - Standard deviation is clamped to prevent numerical instability.
        - Returns only the upper triangular portion corresponding to forward time lags.
    """
    B, N, D = points.size()

    # roll to calculate lagged cov
    stack = torch.concat(
        [torch.roll(points, x, dims=1) for x in range(max_lags + 1)], dim=2
    )

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1)

    # Ensure std is not too small to avoid NaNs or Infs
    std = std.clamp(min=eps)

    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1))

    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate

    # make correlation out of it by dividing by the product of the stds
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    ).clamp(min=eps)  # Add small epsilon to denominator directly

    # we can remove backwards in time links (keep only the original values)
    return corr[:, :D, D:]  # shape: (B, D, D)


def adaptive_threshold_regularization(predictions: torch.tensor, data: torch.tensor, percentile: int=75) -> torch.tensor:
    """
    Penalizes mismatch with crosscorrelation (CC) signal, based on a percentile threshold.
    May be used to reduce underprediction on strong CC entries.
    """
    max_lag = predictions.shape[3]
    n_vars = data.shape[2]

    corr = lagged_batch_crosscorrelation(data, max_lag)
    fncorr = transform_corr_to_y(corr, max_lag, n_vars)

    threshold = torch.percentile(fncorr, percentile)
    mask = fncorr > threshold
    penalty = torch.mean((predictions * mask) ** 2)  # Penalize low correlations (below threshold)

    return penalty


def corr_regularization(predictions: torch.tensor, data: torch.tensor, exp: float=1.5, epsilon: float=1e-8) -> torch.tensor:
    """
    Penalize mismatch with crosscorrelation (CC) signal, emphasizing underprediction on strong CC entries.

    Args
        - predictions (torch.tensor): Predictions tensor of shape `[batch_size, num_vars, num_vars, max_lag]`
        - data (torch.tensor): Data tensor of shape `[batch_size, timesteps, num_vars]`
        - epsilon (float): Small value to avoid division by zero; default is `1e-8`.

    Returns
        - loss (torch.tensor): Regularization loss
    """
    max_lag = predictions.shape[3]
    n_vars = data.shape[2]

    corr = lagged_batch_crosscorrelation(data, max_lag)
    y_corr = transform_corr_to_y(corr, max_lag, n_vars)

    # normalize
    norm_corr = (torch.abs(y_corr) - torch.abs(y_corr).min()) / (torch.abs(y_corr).max() - torch.abs(y_corr).min() + epsilon)

    loss = torch.mean(((predictions - norm_corr) ** 2) * norm_corr**exp) # penalize under-over with mean weighted squared error

    return loss


def read_to_csv(data_path: Path, column_names: list=None) -> pd.DataFrame:
    """ 
    A general utility method that reads from various data types and returns the corresponding pandas.DataFrame object. 

    Args
    ----
    - data_path (pathlib.Path) : the path to the data
    - column_names (list) : a list containing the names of the columns to be assigned to the data 

    Return
    ------
    - data_pd (pandas.DataFrame) : the data as a pandas.DataFrame object
    """
    if column_names is None:
        column_names = list(string.ascii_uppercase) + ["".join(a) 
                                                       for a in list(itertools.permutations(list(string.ascii_uppercase), r=2))]
    
    postfix = os.path.basename(data_path) 
    
    if ".csv" in postfix: # for data in .csv format
        true_data = pd.read_csv(data_path)

    elif ".npy" in postfix: # for data in .npy format
        true_data = pd.DataFrame(data=np.load(data_path))

    elif ".txt" in postfix: # for data in .txt format
        true_data = pd.read_csv(data_path, sep=" ", header=None)

    else:
        raise ValueError("Unsupported data format.")

    # These may be redundant - trying to solve some incompatibilities
    true_data = true_data.rename(columns=dict(zip(true_data.columns, column_names[:true_data.shape[1]])))
    true_data = true_data.dropna(axis=0)
    # true_data = true_data.astype('float')

    return true_data

def check_non_stationarity(df: pd.DataFrame, verbose: bool=False) -> bool:
    """
    Given a time-series sample, checks for non-stationarity using the Augmented Dickey-Fuller test.
    
    Args:
       df (pd.DataFrame) : multivariate time-series sample of shape `(n,d)` where `n` is the sample size and `d` the feature size 
       verbose (bool) : whether to print which feature is non-stationary (default: False)
    
    Returns:
       bool
       `True` if there exists a non-stationary feature, `False` otherwise.    
    """
    # Hyperparameters
    a_fuller = 0.05
    a_kolmogorov = 0.05

    stat_and_p = {}

    # 1. Per column checks
    for col in df.columns:
        ## 1.1 Check if time-series are stationary
        adf, pvalue, used_lag, _, _, _ = adfuller(df.loc[:, [col]].values)

        if pvalue>a_fuller: 
            if verbose:
                print(f"Time-series corresponding to variable {col} are not stationary.")
            return True
    return False 

def to_stationary_with_finite_differences(df: pd.DataFrame, order: int=1) -> pd.DataFrame:
    """
    Converts the given (non-stationary) time-series sample to stationary using finite differences of order `order`.

    Args:
       df (pd.DataFrame): multivariate time-series sample of shape `(n,d)` where `n` is the sample size and `d` the feature size,
         where at least one feature is non-stationary.
       order (int): order of finite differences to take (default: `1`)

    Returns:
       pd.DataFrame
        Finite-differenced pandas DataFrame of the non-stationary multivariate time-series
    """
    if check_non_stationarity(df) == False:
        warnings.warn("Provided time-series sample is stationary. No finite differencing is applied.")
        return df

    diff_df = df.diff(periods=order).dropna().reset_index(drop=True)

    return diff_df

def convert_data_to_stationary(df: pd.DataFrame, order: int=1, verbose=False) -> pd.DataFrame:
    """
    Converts a dataset containing non-stationary features to a stationary one using finite differences. In case all features
    are stationary, the dataset is not modified and returned by the method as it is.

    Args:

       df (pd.DataFrame): The data sample of shape `(n,d)` where `n` is the sample size and `d` the feature size
       order (int): The order to account for in the finite-differences method (default: 1)
       verbose (bool): Whether to print process messages (default: `False`).

    Returns:

       pd.DataFrame
         The stationary-transformed dataset    
    """
    if check_non_stationarity(df, verbose=verbose):
        diff_df = to_stationary_with_finite_differences(df, order=order)
        return diff_df
    return df


def compute_roc_metrics(predictions, labels):
    """
    Compute ROC and AUROC metrics from predictions and labels.

    Args:
        predictions (Tensor): Model output probabilities or scores.
        labels (Tensor): Ground truth binary labels.

    Returns:
        Tuple: ROC curve and AUROC score.
    """
    
    predictions = torch.Tensor(predictions)
    labels = labels.type(torch.int64)

    roc_metric = torchmetrics.classification.BinaryROC()
    auroc_metric = torchmetrics.classification.BinaryAUROC()

    roc_curve = roc_metric(preds=predictions, target=labels)
    auroc_score = auroc_metric(preds=predictions, target=labels)

    return roc_curve, auroc_score


def run_varlingam_on_sample(sample, max_lag=3, threshold=1e-5):
    """
    Run VARLiNGAM and construct lagged adjacency tensor [j, i, l]
    where i → j with lag = max_lag - l (matching label convention)
    """
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()

    model = VARLiNGAM(lags=max_lag)
    model.fit(sample)
    coef_matrices = model.adjacency_matrices_

    D = coef_matrices[0].shape[0]
    lagged_adj = np.zeros((D, D, max_lag))  # shape [j, i, l]

    for lag, matrix in enumerate(coef_matrices):
        lag_l = max_lag - (lag + 1)  # match label convention
        if lag_l < 0 or lag_l >= max_lag:
            continue
        lagged_adj[:, :, lag_l] = (np.abs(matrix.T) > threshold).astype(int)  # j → i flipped to i → j

    return lagged_adj

def varlingam_score_matrix(sample, max_lag=3):
    """
    Return absolute coefficient-based lagged adjacency tensor
    with shape [j, i, l] meaning i → j with lag = max_lag - l
    """
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()

    model = VARLiNGAM(lags=max_lag)
    model.fit(sample)
    coef_matrices = model.adjacency_matrices_

    D = coef_matrices[0].shape[0]
    lagged_score_adj = np.zeros((D, D, max_lag))

    for lag, matrix in enumerate(coef_matrices):
        lag_l = max_lag - (lag + 1)
        if lag_l < 0 or lag_l >= max_lag:
            continue
        lagged_score_adj[:, :, lag_l] = np.abs(matrix.T)  # Keep absolute value, no threshold

    return lagged_score_adj


def run_varlingam_on_dataset(dataset, max_lag=3):
    results = []
    labels = []

    for data_sample, ground_truth in dataset:
        inferred_graph = run_varlingam_on_sample(data_sample, max_lag=max_lag)
        results.append(inferred_graph)
        labels.append(ground_truth)

    return np.stack(results, axis=0), np.stack(labels, axis=0)


def run_varlingam_with_bootstrap(sample, max_lag, n_sampling=100, min_causal_effect=0.05):

    model = VARLiNGAM(lags=max_lag)

    if isinstance(sample, torch.Tensor):
        sample_np = sample.detach().cpu().numpy()
    elif isinstance(sample, np.ndarray):
        sample_np = sample
    else:
        raise ValueError("Sample must be torch.Tensor or np.ndarray")

    n_vars = sample_np.shape[1]

    result = model.bootstrap(sample_np, n_sampling=n_sampling)
    prob_matrices = result.get_probabilities(min_causal_effect=min_causal_effect)

    assert len(prob_matrices) == max_lag + 1, "Mismatch in number of lags."

    prob_adj = np.zeros(shape=(n_vars, n_vars, max_lag))
    for lag in range(max_lag):
        varlingam_lag = max_lag - lag
        if varlingam_lag > max_lag:
            continue # invalid
        prob_adj[:, :, lag] = prob_matrices[varlingam_lag]

    return prob_adj


def compute_auc_from_varlingam(score_adj, label):
    """
    Compute AUC based on score matrix (absolute coefficients)
    and ground-truth label tensor.
    """
    # Convert to tensors if not already
    score_adj = torch.tensor(score_adj) if not isinstance(score_adj, torch.Tensor) else score_adj
    label = torch.tensor(label) if not isinstance(label, torch.Tensor) else label

    # Mask self-edges
    mask = torch.ones_like(label, dtype=bool)
    for d in range(label.shape[0]):
        mask[d, d, :] = False

    flat_label = (label[mask].flatten() > 0.05).int()  # Binary labels
    flat_score = score_adj[mask].flatten().float()     # Continuous scores

    # Handle degenerate case
    if (flat_label == 0).all() or (flat_label == 1).all():
        return torch.nan

    # Torchmetrics AUROC
    auc_metric = BinaryAUROC()
    auc = auc_metric(flat_score, flat_label)

    return auc.item()