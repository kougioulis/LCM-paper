import sys
sys.path.append(".")

import numpy as np
import pandas as pd
import torch
import torchmetrics
from tigramite import data_processing as pp
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr_wls import ParCorr
from tigramite.pcmci import PCMCI
from src.utils.transformation_utils import to_lagged_adj_ready
from src.utils.utils import compute_roc_metrics, timing


def tensor_to_pcmci_res_modified(sample: torch.tensor, c_test: str, max_tau: int) -> np.array:
    """
    Converts a time-series sample tensor (time-series dataset of shape `(n_vars, max_tau)`)
    to an appropriate dataframe for PCMCI and then runs the PCMCI algorithm.

    Args:
        sample (torch.tensor): the sample tensor described before
        c_test (str): cond ind test to be used ("ParCorr" or "GPDC").
        max_tau (int): maximum lag
    
    Returns:
        the PCMCI q-matrix 
    """

    if c_test == "ParCorr":
        c_test = ParCorr()
    elif c_test == "GPDC":
        c_test = GPDC()
    else:
        raise Exception("c_test must be either ParCorr or GPDC")

    dataframe = pp.DataFrame(
        sample.detach().numpy().astype(float),
        datatime=np.arange(len(sample)),  # time-axis for PCMCI
        var_names=np.arange(sample.shape[1])  # should be (0,1,..., num_vars-1)
    )
    
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=0)
    results = pcmci.run_pcmci(tau_max=max_tau) # p-values output of shape `(num_vars, num_vars, max_tau + 1)`

    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"], fdr_method="fdr_bh"
    ) # returns the corrected p-values using the Benjamini-Hochberg method, including contemporaneous edges with shape `(num_vars, num_vars, max_tau +1)`

    q_matrix = np.swapaxes(np.flip(q_matrix[:, :, 1:], 2), 0, 1)

    return q_matrix


def run_inv_pcmci(sample: pd.DataFrame, c_test=None, max_tau: int=1, fdr_method: str="fdr_bh", invert: bool=True, rnd: int=3, threshold: float=0.05) -> np.array:
    """
    Converts an fMRI datasample to appropriate dataframe for PCMCI and then runs the PCMCI algorithm.

    Args:
        sample (pd.DataFrame) : the time-series data as a Pandas DataFrame 
        c_test (tigramite.independence_tests) : conditional independence test to be used (ParCorr() or GPDC())
        max_tau (int) : (optional) the maximum lag to use; default is `1`.
        fdr_method (str) : (optional) the FDR method that PCMCI will use internally; for more info, 
                           please refer to the official PCMCI documentation
        invert (bool) : (optional) if true, it inverts the time-slices of the returning adjacency matrix, 
                           in order to match the effect-case order of CP
        rnd (str): (optional) the rounding range for the output 
        threshold (float) : (optional) the threshold on which the corrected p-values of the p-matrix are adjusted; default is `0.05`.
    
    Returns:
        the PCMCI q-matrix (numpy.array) without contemporaneous edges, of shape `(num_vars, num_vars, max_tau)`
    """
    if isinstance(sample, pd.DataFrame):
        sample = sample.values
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().numpy()

    if c_test is None:
        c_test = ParCorr()

    dataframe = pp.DataFrame(
        sample,
        datatime=np.arange(sample.shape[0]),
        var_names=np.arange(sample.shape[1]),
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=1)
    pcmci.verbosity = 0
    results = pcmci.run_pcmci(tau_min=0, tau_max=max_tau, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues( # returns corrected p-values
        p_matrix=results["p_matrix"], fdr_method=fdr_method
    )
    out = q_matrix[:, :, 1:] # exclude contamporaneous edges

    # Select the edges with low p-values, zero-out the edges with high-p-values
    out[out < threshold] = 1
    out[out < 1] = 0

    if invert:
        out = to_lagged_adj_ready(out.round(rnd))
    else:
        out = torch.tensor(out)
    
    return out

def run_inv_pcmciplus(sample: pd.DataFrame, c_test=None, max_tau: int=1, fdr_method: str="fdr_bh", invert: bool=True, rnd: int=3, threshold: float=0.05) -> np.array:
    """
    Converts a datasample to an appropriate dataframe for PCMCI+ and then runs the PCMCI+ algorithm.

    Args:
        sample (pd.DataFrame) : the time-series data as a Pandas DataFrame 
        c_test (tigramite.independence_tests) : conditional independence test to be used (ParCorr() or GPDC())
        max_tau (int) : (optional) the maximum lag to use; defaults to 1
        fdr_method (str) : (optional) the FDR method that PCMCI will use internally; for more info, 
                           please refer to the official PCMCI documentation
        invert (bool) : (optional) if true, it inverts the time-slices of the returning adjacency matrix, 
                           in order to match the effect-case order of CP
        rnd (str): (optional) the rounding range for the output 
        threshold (float) : (optional) the threshold on which the corrected p-values of the p-matrix are adjusted; default is 0.05
    
    Returns:
        the PCMCI+ q-matrix (numpy.array) excluding contemporaneous edges, of shape `(num_vars, num_vars, max_tau)`
    """
    if isinstance(sample, pd.DataFrame):
        sample = sample.values
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().numpy()

    if c_test is None:
        c_test = ParCorr()

    dataframe = pp.DataFrame(
        sample,
        datatime=np.arange(sample.shape[0]),
        var_names=np.arange(sample.shape[1]),
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=1)
    pcmci.verbosity = 0
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=max_tau, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"], fdr_method="fdr_bh", exclude_contemporaneous=True # excluding cont. edges as PCMCI+ accouts for that
    )         
    out = q_matrix[:, :, 1:] # exclude contamporaneous edges

    # Select the edges with low p-values, zero-out the edges with high-p-values
    out[out < threshold] = 1
    out[out < 1] = 0

    if invert:
        out = to_lagged_adj_ready(out.round(rnd))
    else:
        out = torch.tensor(out)
    
    return out


def run_pcmci_on_sample(sample: torch.Tensor, cond_test: str, max_lag: int=1) -> np.ndarray:
    """
    Run PCMCI on a single time-series sample.

    Args:
        sample (Tensor): Time-series sample (T x D).
        cond_test (str): Conditional independence test object (e.g., `"ParCorr"`).
        max_lag (int): Maximum time lag (default is `1`).

    Returns:
        np.ndarray: Corrected q-value matrix from PCMCI.
    """
    if isinstance(sample, (tuple, list)):
        sample = torch.stack(sample) if isinstance(sample[0], torch.Tensor) else torch.tensor(sample)
    elif not isinstance(sample, torch.Tensor):
        raise ValueError(f"Unexpected sample type: {type(sample)}")

    # Normalize data
    sample = (sample - sample.mean(dim=0)) / (sample.std(dim=0) + 1e-6)

    dataframe = pp.DataFrame(
        sample.detach().numpy().astype(float),
        datatime=np.arange(len(sample)),
        var_names=np.arange(sample.shape[1]),
    )

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_test, verbosity=0)
    results = pcmci.run_pcmci(tau_min=0, tau_max=max_lag, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"], fdr_method="fdr_bh"
    )
    return q_matrix[:,:,1:]  # exclude contemporaneous edges


@timing
def run_pcmci_on_dataset(dataset, cond_test: str = "ParCorr", max_lag: int = 3):
    """
    Apply PCMCI to an entire dataset.

    Args:
        dataset (Iterable[Tuple[Tensor, np.ndarray]]): List of (time-series, ground truth) tuples.
        cond_test: Conditional independence test object.
        max_lag (int): Maximum lag for PCMCI.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted q-values and ground truth graphs.
    """
    if cond_test == "GPDC":
        cond_test = GPDC()
    elif cond_test == "ParCorr":
        cond_test = ParCorr()
    else:
        raise ValueError(f"Unsupported test type: {cond_test}")

    results = []
    labels = []

    for data_sample, ground_truth in dataset:
        q_matrix = run_pcmci_on_sample(data_sample, cond_test, max_lag=max_lag)
        lagged_q = np.swapaxes(np.flip(q_matrix[:, :, 1:], axis=2), 0, 1)
        results.append(lagged_q)
        labels.append(ground_truth)

    return np.stack(results, axis=0), np.stack(labels, axis=0)


def apply_pcmci_to_dataloader(dataloader, test_type="GPDC"):
    """
    Apply PCMCI to a dataloader containing time-series batches.

    Args:
        dataloader (Iterable[Tuple[Tensor, Tensor]]): Iterable over `(x, y)` batches.
        test_type (str): Conditional test (`"GPDC"` or `"ParCorr"`).

    Returns:
        Tuple[Tensor, Tensor]: Predicted q-values and ground truth labels.
    """
    cond_test = GPDC() if test_type == "GPDC" else ParCorr() if test_type == "ParCorr" else None
    if cond_test is None:
        raise ValueError(f"Unsupported test type: {test_type}")

    all_preds, all_labels = [], []

    for x_batch, y_batch in dataloader:
        preds, _ = run_pcmci_on_dataset(x_batch, cond_test, max_lag=y_batch.shape[3])
        all_preds.append(preds)
        all_labels.append(y_batch)

    return torch.Tensor(np.concatenate(all_preds)), torch.concat(all_labels)


def evaluate_pcmci_direction_accuracy(data):
    """
    Evaluate directionality accuracy of PCMCI between two variables.

    Args:
        data (Tuple[Tensor, Tensor]): Tuple of (data, labels).

    Returns:
        float: Proportion of samples where correct direction is stronger.
    """
    cond_test = ParCorr()
    x, _ = data
    results, _ = run_pcmci_on_dataset(x, cond_test, max_lag=1)

    results = torch.Tensor(results)
    direction_correct = (
        results[:, 0, 1].max(dim=1)[0] > results[:, 1, 0].max(dim=1)[0]
    ).sum()

    return direction_correct.item() / len(results)


def compute_pcmci_roc_without_diagonal(dataloader, max_lag=1, num_vars=15):
    """
    Compute ROC/AUROC after masking diagonal self-dependencies.

    Args:
        dataloader (Iterable[Tuple[Tensor, Tensor]]): Data batches.
        max_lag (int): Max lag to use in PCMCI.
        num_vars (int): Number of variables in dataset.

    Returns:
        Tuple: ROC curve and AUROC score.
    """
    roc_metric = torchmetrics.classification.BinaryROC()
    auroc_metric = torchmetrics.classification.BinaryAUROC()
    cond_test = ParCorr()

    preds_all, labels_all = [], []
    for x_batch, y_batch in dataloader:
        preds, _ = run_pcmci_on_dataset(x_batch, cond_test, max_lag=max_lag)
        preds_all.append(torch.Tensor(preds))
        labels_all.append(torch.Tensor(y_batch))

    preds = torch.concat(preds_all, dim=0)
    labels = torch.concat(labels_all, dim=0)

    mask = ~torch.eye(num_vars, num_vars).flatten().bool()

    def flatten_and_mask(batch):
        return [x[:, :, 0].flatten()[mask] for x in batch]

    masked_preds = torch.concat(flatten_and_mask(preds), dim=0)
    masked_labels = torch.concat(flatten_and_mask(labels), dim=0)

    return compute_roc_metrics(masked_preds, masked_labels, roc_metric, auroc_metric)