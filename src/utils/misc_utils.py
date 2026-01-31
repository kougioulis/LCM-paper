import random
import sys
import re
import os
import time
import numpy as np
import pandas as pd
import torch
import pickle 
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
from prettytable import PrettyTable  # used by count_params() function
from functools import wraps
from typing import Union
from itertools import combinations
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve
from collections import defaultdict
import warnings

from IPython.display import display

sys.path.append("..")
from src.utils.metrics import custom_binary_metrics
from src.utils.pcmci_utils import tensor_to_pcmci_res_modified
from src.utils.dynotears_utils import run_dynotears_with_bootstrap
from src.utils.cdml_utils import y_from_cdml_to_lagged_adj
from src.utils.transformation_utils import from_fmri_to_lagged_adj
from src.utils.utils import check_non_stationarity, to_stationary_with_finite_differences, lagged_batch_crosscorrelation, \
    run_varlingam_with_bootstrap

from src.utils.plotting_utils import plot_comparison_fancy

def timing(f: callable) -> callable:
    """
    Just a timing decorator.
    """
    @wraps(f)
    def wrap(*args, **kw):
        tic = time()
        result = f(*args, **kw)
        tac = time()
        print("Elapsed time: %2.4f seconds" % (tac - tic))

        return result, tac - tic
    

def print_time_slices(adj: Union[torch.tensor, np.ndarray]) -> None:
    """
    Simply print the time slices of a lagged adjacency matrix. Shape of the matrix should be `(n_vars, n_vars, n_lags)`. 

    Args: 
        adj (torch.Tensor or numpy.array): 
    """
    for t in range(adj.shape[2]):
        print(adj[:, :, t])


def extract_number(filename: str, pattern: str) -> int:
    """ 
    Extracts the number used to describe a unique file. 
    Used specifically when reading files from the fMRI dataset collection.

    Args: 
        filename (str): The name of the file
        pattern (str): The regex pattern used to extract the number
    
    Returns (int):
        The integer identifier
    """
    match = re.search(pattern, filename)

    return int(match.group(1)) if match else None

def fmri_to_adjacency_tensor(test_fmri: torch.Tensor, label_fmri: torch.Tensor, max_lag: int=1):
    """
    Constructs a lagged adjacency tensor from an instance of the fMRI dataset.

    Args:
        test_fmri (torch.Tensor): the time-series data (used for extracting variable size)
        label_fmri (torch.Tensor): the ground truth label (causal graph)
        max_lag (torch.Tensor) : The maximum lag (delay)

    Returns (torch.Tensor):
        the fMRI label to a lagged adjacency tensor format
    """
    # Construct time-lagged adj matrix
    Y_fmri = np.zeros(shape=(test_fmri.shape[1], test_fmri.shape[1], max_lag)) # (dim, dim, time)

    for idx in label_fmri.index:
        Y_fmri[label_fmri['effect'], label_fmri['cause'], max_lag-label_fmri['delay']] = 1
    Y_fmri = torch.tensor(Y_fmri)

    return Y_fmri


def get_fmri_pairs(timeseries_files: list[str], ground_truth_files: list[str], verbose=False) -> list[tuple[str, str]]:
    """
    Helper to extract time-series and ground truth causal graph pairs of lists of time-series and causal graphs.
    Uses regular expressions to create a 1-1 correspondence between the two.

    Args:
        timeseries_files (list): List of strings for fMRI sample files
        ground_truth_files (list): List of strings for fMRI causal graph files
        verbose (bool): Whether to enable verbose logging. Default is `False`.

    Returns (list):
        List of string tuples of the form (time series sample, ground truth graph)
    """

    # regex patterns to extract numbers from filenames
    timeseries_pattern = r'timeseries(\d+)\.csv'
    ground_truth_pattern = r'sim(\d+)_gt_processed\.csv'
    matched_files = []

    for ts_file in timeseries_files:
        ts_number = extract_number(ts_file, timeseries_pattern)
        for gt_file in ground_truth_files:
            gt_number = extract_number(gt_file, ground_truth_pattern)
            if ts_number == gt_number:
                matched_files.append((ts_file, gt_file))

    if verbose:
        for ts_file, gt_file in matched_files:
            print(f"Timeseries file: {ts_file} -> Ground truth file: {gt_file}")

    return matched_files


def count_params(model: torch.nn.Module, pretty: bool=False) -> int:
    """
    Counts the number of (trainable) parameters of a `torch.nn.Module` model.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model
        pretty (bool): Whether to enable pretty view of parameters (using PrettyTable). Default is `False`.

    Returns (int):
        Total number of (trainable) parameters
    """
    if pretty:
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")

        return total_params

    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_aligned_aucs(model_data_1, model_data_2):
    """
    Align per-sample AUCs from two models.

    Expects each input to be an iterable of (sample_id, auc).
    """
    d1 = dict(model_data_1)
    d2 = dict(model_data_2)

    common_keys = sorted(d1.keys() & d2.keys())
    if len(common_keys) == 0:
        raise ValueError("No shared samples between models.")

    scores_a = [d1[k] for k in common_keys]
    scores_b = [d2[k] for k in common_keys]
    return scores_a, scores_b


def aggregate_across_runs(per_sample_results):
    aggregated = {}
    for model, pairs in per_sample_results.items():
        tmp = defaultdict(list)
        for sample_id, auc in pairs:
            tmp[sample_id].append(auc)
        aggregated[model] = [
            (sample_id, float(np.mean(aucs)))
            for sample_id, aucs in tmp.items()
        ]
    return aggregated


def perform_wilcoxon_test(
    per_sample_results: dict,
    metric: str = "AUC",
    adjust_for_multiple_tests: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Performs pairwise Wilcoxon signed-rank tests on aligned per-sample metrics.

    Returns:
        pd.DataFrame with pairwise test results.
    """
    model_pairs = list(combinations(per_sample_results.keys(), 2))
    if not model_pairs:
        raise ValueError("No model pairs found.")

    adjusted_alpha = (
        alpha / len(model_pairs) if adjust_for_multiple_tests else alpha
    )

    results = []

    for model_a, model_b in model_pairs:
        if metric != "AUC":
            raise NotImplementedError(f"Metric {metric} not supported.")

        scores_a, scores_b = get_aligned_aucs(
            per_sample_results[model_a],
            per_sample_results[model_b],
        )

        if len(scores_a) < 5:
            p_val = np.nan
        else:
            try:
                stat, p_val = wilcoxon(
                    scores_a,
                    scores_b,
                    zero_method="pratt",
                    alternative="two-sided",
                )
            except ValueError:
                p_val = np.nan

        results.append({
            "model_a": model_a,
            "model_b": model_b,
            "n_shared_samples": len(scores_a),
            "mean_a ± std": f"{np.mean(scores_a):.4f} ± {np.std(scores_a, ddof=1):.4f}",
            "mean_b ± std": f"{np.mean(scores_b):.4f} ± {np.std(scores_b, ddof=1):.4f}",
            "raw_p_value": p_val,
            "adjusted_alpha": adjusted_alpha,
            "significant_after_correction": (
                p_val < adjusted_alpha if not np.isnan(p_val) else False
            ),
        })

    return pd.DataFrame(results)



def load_sharded_dataset(base_path: Path, split: str): 
    """
    Loads all torch shards (e.g., test_merged_shard0.pt, test_merged_shard1.pt etc)
    and yields them one by one to avoid loading everything in memory.

    Args:
        base_path (Path): Path to the folder containing the pytorch .pt shards in the format `<split>_shard*.pt`.
        split (str): Name of the split to load.
    """
    split_path = base_path / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    shard_files = sorted(split_path.glob(f"{split}_merged_shard*.pt"))
    if not shard_files:
        print("Trying naming convention")
        shard_files = sorted(split_path.glob(f"{split}_shard*.pt"))
    if not shard_files:
        print(f"No shards found, trying single file {split}.pt")
        single_file = split_path / f"{split}.pt"
        if single_file.exists():
            yield torch.load(single_file)
            return
        else:
            raise FileNotFoundError(f"No dataset found for split {split}")

    print(f"- Found {len(shard_files)} shard(s) for split '{split}'")

    for shard_idx, shard_path in enumerate(shard_files):
        shard_data = torch.load(shard_path, weights_only=False)
        print(f"  ├── Loaded {shard_path.name:35s} ({len(shard_data)} datasamples)")
        yield shard_data

    print(f"Shards have been processed.\n")


def load_full_dataset(base_path: Path, split: str) -> list:
    """
    Loads a single dataset file into memory.

    Args:
        base_path (Path): Path to the folder containing the pytorch .pt shards in the format `<split>_shard*.pt`.
        split (str): Name of the split to load.

    Returns:
        list: List of data samples (pairs of time series and ground truth lagged causal graph).
    """
    file_path = base_path / f"{split}.pt"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    data = torch.load(file_path, weights_only=False)
    print(f"Loaded dataset file: {file_path} ({len(data)} samples)")

    return data


def run_evaluation_experiments(models: dict, cpd_path: Path, out_dir: Path,
                               split: str="test",
                               sharded_data: bool=False,
                               fmri_data: bool=False,
                               kuramoto_data: bool=False,
                               dataset_label: str=None,
                               N_SAMPLING: int=10, N_RUNS: int=5,
                               MAX_VAR: int=12, MAX_LAG: int=3) -> None:
    """
    Run evaluation experiments for multiple causal models on a given dataset.

    Args:
        models (dict): dict of model_name -> model (None for non-neural methods)
        cpd_path (Path): path to causal dataset directory
        out_dir (Path): output directory for results
        split (str): The data split to use
        sharded_data (bool): Whether data to be loaded are sharded; default is False
        fmri_data (bool): Whether to evaluate on fMRI data collections; default is False
        kuramoto_data (bool): Whether to evaluate on Kuramoto data collections; default is False
        dataset_label (str): Label for saving results; default is None - points to the folder stem
        N_SAMPLING (int): number of bootstrapped samples; default is 10
        N_RUNS (int): number of repeated runs for standard error estimation; default is 5 
        MAX_VAR (int): maximum number of variables to consider; default is 12
        MAX_LAG (int): maximum number of lags to consider; default is 3

    Returns:
        None
    """

    """ Path """
    cpd_path = Path(cpd_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    """ Placeholders """
    results_df = pd.DataFrame(columns=[
        "model", "AUC_mean", "AUC_se", "TPR_mean", "TPR_se", "FPR_mean", "FPR_se", 
        "TNR_mean", "TNR_se", "FNR_mean", "FNR_se", "Precision_mean", "Precision_se", 
        "Recall_mean", "Recall_se", "F1_mean", "F1_se"
    ])
    results_dict = {}
    per_sample_results = defaultdict(list)
    running_times_dict = defaultdict(list)

    print(f'Dataset: {Path(cpd_path).parent.stem}')

    """ Load data depending on mode """
    if fmri_data:
        fmri_path = Path(cpd_path)

        print(f"Using fMRI dataset from: {fmri_path}")
        timeseries_files = [f for f in os.listdir(fmri_path) if 'timeseries' in f]
        ground_truth_files = [f for f in os.listdir(fmri_path) if 'gt_processed' in f]

        matched_files = []
        for ts_file in timeseries_files:
            ts_number = extract_number(ts_file, r'timeseries(\d+)\.csv')
            for gt_file in ground_truth_files:
                gt_number = extract_number(gt_file, r'sim(\d+)_gt_processed\.csv')
                if ts_number == gt_number:
                    matched_files.append((ts_file, gt_file))
        print(f"Total number of fMRI data pairs: {len(matched_files)}")

        dataset_iterator = []
        for ts_file, gt_file in matched_files:
            test_fmri = pd.read_csv(f"{fmri_path}/{ts_file}")
            label_fmri = pd.read_csv(f"{fmri_path}/{gt_file}", names=['effect', 'cause', 'delay'])

            X_fmri = torch.tensor(test_fmri.values, device='cpu', dtype=torch.float32)
            Y_fmri = from_fmri_to_lagged_adj(test_fmri=test_fmri, label_fmri=label_fmri)

            # Skip degenerate samples
            if Y_fmri.sum() <= 0 or Y_fmri.sum() == np.prod(Y_fmri.shape):
                continue
            
            dataset_iterator.append([(X_fmri, Y_fmri)]) # list of datasamples

    else:
        #if cpd_path.suffix in [".p", ".pkl", ".pickle"]:
        #    print(f"--PICKLE mode-- ({cpd_path.name})")
        #    with open(cpd_path, "rb") as f:
        #        data = pickle.load(f)
        if sharded_data:
            print("--SHARDED mode--")
            dataset_iterator = load_sharded_dataset(cpd_path, split)
        else:
            print("--SINGLE FILE mode--")
            dataset_iterator = [load_full_dataset(cpd_path, split)]

    """ Model loop """
    for model_name, model in zip(models.keys(), models.values()):

        print(f"\n___{model_name}___")

        MAX_VAR = MAX_VAR 
        MAX_LAG = MAX_LAG
        if "CP_trf" in model_name:
            MAX_VAR = 5
        elif "LCM" in model_name:
            MAX_VAR, MAX_LAG = 12, 3

        if fmri_data:
            if ("PCMCI" in model_name) or ("DYNOTEARS" in model_name) or ("VARLINGAM" in model_name):
                if "fMRI_5" in str(cpd_path):
                    MAX_VAR = 5
                    MAX_LAG = label_fmri['delay'].max()
                else:
                    MAX_VAR = 10
                    MAX_LAG = label_fmri['delay'].max()
            elif "CP_trf" in model_name:
                MAX_VAR = 5
                MAX_LAG = 3
            else:
                MAX_VAR = MAX_VAR 
                MAX_LAG = MAX_LAG 

        if kuramoto_data:
            if ("PCMCI" in model_name) or ("DYNOTEARS" in model_name) or ("VARLINGAM" in model_name):
                if "kuramoto_5" in str(cpd_path):
                    MAX_VAR = 5
                    MAX_LAG = 1 
                else:
                    MAX_VAR = 10
                    MAX_LAG = 1 
            elif "CP_trf" in model_name:
                MAX_VAR = 5
                MAX_LAG = 3
            else:
                MAX_VAR = MAX_VAR 
                MAX_LAG = MAX_LAG
        
        print(f"VAR: {MAX_VAR} | MAX LAG: {MAX_LAG}") if "LCM" in model_name else print(f"MAX LAG: {MAX_LAG}") 

        # store metrics across multiple runs
        run_metrics = {"AUC": [], "TPR": [], "FPR": [], "TNR": [], "FNR": [], "Precision": [], "Recall": [], "F1": []}

        # PCMCI is deterministic, multiple runs are not necessary, while DYNOTEARS and VARLINGAM are effectively deterministic (se < 1e-4)
        if model_name == "PCMCI" or model_name == "DYNOTEARS" or model_name == "VARLINGAM":
            N_RUNS = 1

        for run_id in range(N_RUNS):
        #    print(f"\n--- Run {run_id+1}/{N_RUNS} ---")
            seed = 42 + run_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            print(f"\n |--- Run {run_id+1}/{N_RUNS} (seed={seed}) ---")

            # Re-initializing the dataset iterator each run
            if fmri_data:
                dataset_iterator = [] 
                for ts_file, gt_file in matched_files:
                    test_fmri = pd.read_csv(f"{fmri_path}/{ts_file}")
                    label_fmri = pd.read_csv(f"{fmri_path}/{gt_file}", names=['effect', 'cause', 'delay'])
                    X_fmri = torch.tensor(test_fmri.values, device='cpu', dtype=torch.float32)
                    Y_fmri = from_fmri_to_lagged_adj(test_fmri=test_fmri, label_fmri=label_fmri)
                    if Y_fmri.sum() <= 0 or Y_fmri.sum() == np.prod(Y_fmri.shape):
                        continue
                    dataset_iterator.append([(X_fmri, Y_fmri)])
            else:
                if sharded_data:
                    dataset_iterator = load_sharded_dataset(cpd_path, split)
                else:
                    dataset_iterator = [load_full_dataset(cpd_path, split)]

            tpr_list, fpr_list, tnr_list, fnr_list, auc_list = [], [], [], [], []
            precision_list, recall_list, f1_list = [], [], []

            for shard_idx, data in enumerate(dataset_iterator):
                #print(f"\n Processing shard {shard_idx} ({len(data)} samples)")
                for idx in tqdm(range(len(data[:2000])), desc=f'Shard {shard_idx}'):
                    
                    try:
                        X_cpd = data[idx][0]
                        Y_cpd = data[idx][-1]

                        if isinstance(Y_cpd, np.ndarray):
                            Y_cpd = torch.from_numpy(Y_cpd)

                        if model_name == "PCMCI":

                            tic = time.time()
                            pcmci_out = tensor_to_pcmci_res_modified(sample=X_cpd.to('cpu'), c_test="ParCorr", max_tau=Y_cpd.shape[-1])
                            tac = time.time()

                            Y_cpd = (Y_cpd >= 0.05).float()

                            tpr, fpr, tnr, fnr, auc = custom_binary_metrics(torch.tensor(1-pcmci_out), A=Y_cpd, verbose=False)

                            precision = tpr / (tpr + fpr) if tpr + fpr != 0 else 0
                            recall = tpr / (tpr + fnr) if tpr + fnr != 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

                        elif model_name == "DYNOTEARS":

                            max_lag = Y_cpd.shape[2]

                            tic = time.time()
                            pred = run_dynotears_with_bootstrap(pd.DataFrame(X_cpd.detach().numpy()), n_lags=max_lag, n_bootstrap=N_SAMPLING)

                            tac = time.time()

                            if Y_cpd.sum() <= 0 or pred.sum() <= 0:
                                continue
                            
                            Y_cpd[Y_cpd < 0.05] = 0
                            Y_cpd[Y_cpd >= 0.05] = 1

                            pred_bin = (pred >= 0.05).astype(int)

                            assert pred_bin.shape == Y_cpd.shape, \
                                f"Shape mismatch: pred={pred_bin.shape}, Y={Y_cpd.shape}"

                            tpr, fpr, tnr, fnr, auc = custom_binary_metrics(binary=pred_bin, A=Y_cpd, verbose=False)

                            precision = tpr / (tpr + fpr) if tpr + fpr != 0 else 0
                            recall = tpr / (tpr + fnr) if tpr + fnr != 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

                        elif model_name == "VARLINGAM":

                            max_lag = Y_cpd.shape[2]

                            if check_non_stationarity(pd.DataFrame(X_cpd.detach().numpy())):
                                X_cpd = to_stationary_with_finite_differences(
                                    pd.DataFrame(X_cpd.detach().numpy()), order=2
                                )
                                X_cpd = torch.from_numpy(X_cpd.values).float()

                            tic = time.time()

                            score_adj = run_varlingam_with_bootstrap(
                                sample=X_cpd,
                                max_lag=max_lag,
                                n_sampling=N_SAMPLING,
                                min_causal_effect=0.05,
                            )
                            tac = time.time()

                            if score_adj.sum() <= 0 or Y_cpd.sum() <= 0:
                                continue

                            binary_adj_fixed = (score_adj >= 0.5).astype(int)
                            Y_cpd = (Y_cpd >= 0.05).float()

                            tpr, fpr, tnr, fnr, auc = custom_binary_metrics(torch.tensor(binary_adj_fixed), Y_cpd, verbose=False)

                            precision = tpr / (tpr + fpr) if tpr + fpr != 0 else 0
                            recall = tpr / (tpr + fnr) if tpr + fnr != 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

                        else: # neural models
                            M = model.model
                            M = M.to("cpu")
                            M = M.eval() 

                            # Normalization
                            X_cpd = (X_cpd - X_cpd.min()) / (X_cpd.max() - X_cpd.min())

                            # Check dimensions to make sure they do not exceed the model's MAX_LAG & MAX_VAR
                            assert X_cpd.shape[1]<=MAX_VAR, \
                                f"ValueError: input time-series have {X_cpd.shape[1]} variables, while the current model supports at most {MAX_VAR}"
                            assert Y_cpd.shape[1]<=MAX_VAR, \
                                f"ValueError: input adjacency matrix has {Y_cpd.shape[1]} nodes, while the current model supports at most {MAX_VAR}"
                            assert Y_cpd.shape[2]<=MAX_LAG, \
                                f"ValueError: input adjacency matrix has {Y_cpd.shape[2]}, while the current model supports at most {MAX_LAG}"

                            # Padding
                            VAR_DIF = MAX_VAR - X_cpd.shape[1]
                            LAG_DIF = MAX_LAG -Y_cpd.shape[2]
                            if X_cpd.shape[1] != MAX_VAR: # if the number of variables is less than the maximum
                                X_cpd = torch.concat(
                                    [X_cpd, torch.normal(0, 0.01, (X_cpd.shape[0], VAR_DIF))], axis=1 # pad with noise
                                )
                                Y_cpd = torch.nn.functional.pad(
                                    Y_cpd, (0, 0, 0, VAR_DIF, 0, VAR_DIF), mode="constant", value=0.0 # pad the adjacency matrix with zeros
                                )
                            if Y_cpd.shape[2] != MAX_LAG: # if the number of lags is less than the maximum
                                Y_cpd = torch.nn.functional.pad(
                                    Y_cpd, (LAG_DIF, 0, 0, 0, 0, 0), mode="constant", value=0.0 # pad the adjacency matrix with zeros
                                )

                            tic = time.time()

                            if (X_cpd.shape[0]>500):
                                X_cpd = X_cpd[:500]
                                if ("LCM" in model_name) or (model_name=="CP_trf"):
                                    pred = torch.sigmoid(M((X_cpd.unsqueeze(0), lagged_batch_crosscorrelation(X_cpd.unsqueeze(0), 3)))[0])
                                else:
                                    pred = torch.sigmoid(M((X_cpd.unsqueeze(0), lagged_batch_crosscorrelation(X_cpd.unsqueeze(0), 3))))
                                pred = pred.unsqueeze(0)

                                #bs_preds = []
                                #batches = [X_cpd[500*icr: 500*(icr+1), :] for icr in range(X_cpd.shape[0]//500)]
                                #if 500*(X_cpd.shape[0]//500) < X_cpd.shape[0]:
                                #    batches.append(X_cpd[500*(X_cpd.shape[0]//500):, :])

                                #if ("LCM" in model_name) or (model_name=="CP_trf"):
                                #    with torch.no_grad():
                                #        bs_preds = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_crosscorrelation(bs.unsqueeze(0), 3)))) for bs in batches]
                                #else:
                                #    with torch.no_grad():
                                #        bs_preds = [torch.sigmoid(M(bs.unsqueeze(0))) for bs in batches]
                                #preds = torch.cat(bs_preds, dim=0)
                                #pred = preds.mean(0)
                                #pred = pred.unsqueeze(0)
                            else:
                                if ("LCM" in model_name) or (model_name=="CP_trf"):    
                                    with torch.no_grad():
                                        pred = torch.sigmoid(M((X_cpd.unsqueeze(0), lagged_batch_crosscorrelation(X_cpd.unsqueeze(0), 3))))
                                else:
                                    with torch.no_grad():
                                        pred = torch.sigmoid(M(X_cpd.unsqueeze(0)))

                            tac = time.time()

                            #pred[pred < 0.05] = 0
                            #pred[pred >= 0.05] = 1
            
                            Y_cpd[Y_cpd < 0.05] = 0
                            Y_cpd[Y_cpd >= 0.05] = 1

                            if Y_cpd.sum()<=0 or pred[0].sum()<=0:
                                continue

                            """ Binary Metrics """
                            tpr, fpr, tnr, fnr, auc = custom_binary_metrics(binary=pred[0], A=Y_cpd, verbose=False)

                            precision = tpr / (tpr + fpr) if tpr + fpr != 0 else 0
                            recall = tpr / (tpr + fnr) if tpr + fnr != 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

                        auc_list.append(float(auc))
                        tpr_list.append(float(tpr)) 
                        fpr_list.append(float(fpr))
                        tnr_list.append(float(tnr))
                        fnr_list.append(float(fnr))
                        precision_list.append(float(precision))
                        recall_list.append(float(recall))
                        f1_list.append(float(f1))

                        #per_sample_results[model_name].append((idx, auc))
                        sample_id = (shard_idx, idx)
                        per_sample_results[model_name].append((sample_id, auc))

                        running_times_dict[model_name].append(round((tac-tic) / N_SAMPLING, 3))
                    
                    except Exception as e:
                        print(f"\n Error in sample {idx} (shard {shard_idx}) for {model_name}: {type(e).__name__} — {e}")
                        continue

                results_dict[model_name] = [token for token in auc_list]
                for k, v in zip(["AUC","TPR","FPR","TNR","FNR","Precision","Recall","F1"],
                                [auc_list,tpr_list,fpr_list,tnr_list,fnr_list,precision_list,recall_list,f1_list]):
                    run_metrics[k].append(v)   # full list, no mean yet

        def mean_std_allruns(x):
            # Flatten all scores across seeds
            flat = np.array([s for run in x for s in run], dtype=float)
            mu = flat.mean()
            std = flat.std()
            return mu, std

        means_ses = {k: mean_std_allruns(v) for k, v in run_metrics.items()} # NOTE: Standard errors are computed across all datasets and seeds, while in the thesis across seeds only

        results_df.loc[len(results_df), :] = [
            model_name,
            means_ses["AUC"][0], means_ses["AUC"][1],
            means_ses["TPR"][0], means_ses["TPR"][1],
            means_ses["FPR"][0], means_ses["FPR"][1],
            means_ses["TNR"][0], means_ses["TNR"][1],
            means_ses["FNR"][0], means_ses["FNR"][1],
            means_ses["Precision"][0], means_ses["Precision"][1],
            means_ses["Recall"][0], means_ses["Recall"][1],
            means_ses["F1"][0], means_ses["F1"][1],
        ]

    display(results_df)

    if dataset_label is None:
        dataset_label = cpd_path.stem
    save_dir = out_dir / dataset_label
    save_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(save_dir / f"results_metrics_{dataset_label}.csv", index=False)
    
    #model_pairs = list(combinations(per_sample_results.keys(), 2))
    #if len(model_pairs) == 0:
    #    raise ValueError("No model pairs found.")

    #for model_a, model_b in model_pairs:
    #    aucs_a, aucs_b = get_aligned_aucs(per_sample_results[model_a], per_sample_results[model_b])

    aggregated_results = aggregate_across_runs(per_sample_results)
    pairwise_df = perform_wilcoxon_test(aggregated_results) 

    display(pairwise_df)
    pairwise_df.to_csv(save_dir / f"pairwise_significance_{dataset_label}.csv", index=False)

    ###### Figure ######
    plt.figure(figsize=(10, 6))
    plt.boxplot(running_times_dict.values(), labels=running_times_dict.keys(), showfliers=True)
    plt.title("Distribution of Running Times")
    plt.ylabel("Avg running time per dataset (in sec)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"running_times_{dataset_label}.png")
    plt.show()
    plt.close()
    
    print(f"Figure saved to: {out_dir / f'running_times_{dataset_label}.png'}")

    ###############

    running_times_df = pd.DataFrame({k: pd.Series(v) for k, v in running_times_dict.items()})
    running_times_summary = running_times_df.aggregate(['mean', 'median', 'std', 'min', 'max']).T
    running_times_summary.index = [
        f"{label} (running time in sec)" for label in running_times_summary.index
    ]
    display(running_times_summary)
    running_times_summary.to_csv(save_dir / f"running_times_summary_{dataset_label}.csv", index=False)

    # save per-sample results dict to .npy
    np.save(save_dir / f"per_sample_results_{dataset_label}.npy", per_sample_results)

    critical_diff_df = pd.DataFrame(columns=["model", "dataset", "accuracy"])

    for model, auc_list in results_dict.items():
        for auc in auc_list:
                critical_diff_df.loc[len(critical_diff_df)] = [model, dataset_label, auc]

    critical_diff_df.to_csv(save_dir / f"critical_diff_df_{dataset_label}.csv", index=False)

    #display(critical_diff_df)

    print(f"\n Results saved to {out_dir}")



def run_cdml_evaluation_experiments(
    models: dict,
    cdml_path: Path,
    out_dir: Path,
    MAX_VAR: int = 12,
    MAX_LAG: int = 3,
    N_RUNS: int = 5,
    N_SAMPLING: int = 10,
):
    """
    Evaluation for CDML data as in run_evaluation_experiments method.
    """

    cdml_path = Path(cdml_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(columns=[
        "model", "AUC_mean", "AUC_se", "TPR_mean", "TPR_se", "FPR_mean", "FPR_se",
        "TNR_mean", "TNR_se", "FNR_mean", "FNR_se", "Precision_mean", "Precision_se",
        "Recall_mean", "Recall_se", "F1_mean", "F1_se"
    ])

    results_dict = {}
    per_sample_results = defaultdict(list)
    running_times_dict = defaultdict(list)

    print(f"Dataset: {Path(cdml_path).parent.stem}")

    filenames = [
        f.split("_data.csv")[0]
        for f in os.listdir(cdml_path)
        if f.endswith("_data.csv")
    ]
    print(f"Found {len(filenames)} CDML samples.")

    for model_name, model in zip(models.keys(), models.values()):

        print(f"\n___ {model_name} ___")

        max_var = MAX_VAR
        max_lag = MAX_LAG

        # Your original logic
        if model_name == "provided-trf-5V":
            max_var = 5
        elif ("deep" in model_name and ("_10_3" in model_name or "_12_3" in model_name)) \
             or ("lcm" in model_name and "_12_3" in model_name):
            max_var = 12
            max_lag = 3

        print(f"VAR: {max_var} | LAG: {max_lag}")

        # Metrics to aggregate over runs
        run_metrics = {
            "AUC": [], "TPR": [], "FPR": [], "TNR": [], "FNR": [],
            "Precision": [], "Recall": [], "F1": []
        }

        # Deterministic models → 1 run
        if model_name in ["PCMCI", "DYNOTEARS", "VARLINGAM"]:
            N_RUNS = 1

        for run_id in range(N_RUNS):

            seed = 42 + run_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            print(f"\n |--- Run {run_id+1}/{N_RUNS} (seed={seed}) ---")

            tpr_list, fpr_list, tnr_list, fnr_list = [], [], [], []
            auc_list = []
            precision_list, recall_list, f1_list = [], [], []

            for filename in tqdm(filenames):

                # Load CDML data
                X_raw = pd.read_csv(cdml_path / f"{filename}_data.csv")
                Y_raw = pd.read_csv(cdml_path / f"{filename}_target.csv", index_col="Unnamed: 0")

                X = torch.tensor(X_raw.values, dtype=torch.float32)
                X = (X - X.min()) / (X.max() - X.min())  # normalize
                Y = y_from_cdml_to_lagged_adj(Y_raw)

                # Skip incompatible samples
                if X.shape[1] > max_var:
                    continue
                if Y.shape[1] > max_var or Y.shape[2] > max_lag:
                    continue

                # Padding
                VAR_DIF = max_var - X.shape[1]
                LAG_DIF = max_lag - Y.shape[2]

                if VAR_DIF > 0:
                    X = torch.cat([X, torch.normal(0, 0.01, (X.shape[0], VAR_DIF))], dim=1)
                    Y = torch.nn.functional.pad(Y, (0, 0, 0, VAR_DIF, 0, VAR_DIF))
                if LAG_DIF > 0:
                    Y = torch.nn.functional.pad(Y, (LAG_DIF, 0, 0, 0, 0, 0))

                tic = time.time()

                if "LCM" in model_name:
                    M = model.model.to("cpu").eval()
                    if X.shape[0] > 500:
                        batches = []
                        for i in range(X.shape[0] // 500):
                            batches.append(X[500 * i: 500 * (i + 1)])
                        if 500 * (X.shape[0] // 500) < X.shape[0]:
                            batches.append(X[500 * (X.shape[0] // 500):])
                        preds = []
                        for bs in batches:
                            bs = bs.unsqueeze(0)
                            with torch.no_grad():
                                preds.append(torch.sigmoid(M((bs, lagged_batch_crosscorrelation(bs, 3)))))
                        pred = torch.cat(preds, dim=0).mean(0).unsqueeze(0)
                    else:
                        with torch.no_grad():
                            pred = torch.sigmoid(
                                M((X.unsqueeze(0), lagged_batch_crosscorrelation(X.unsqueeze(0), 3)))
                            )

                elif model_name == "PCMCI":
                    pred_np = tensor_to_pcmci_res_modified(X, c_test="ParCorr", max_tau=Y.shape[-1])
                    pred = torch.from_numpy(pred_np.copy())

                elif model_name == "DYNOTEARS":
                    pred_np = run_dynotears_with_bootstrap(
                        pd.DataFrame(X.numpy()),
                        n_lags=max_lag,
                        n_bootstrap=N_SAMPLING
                    )
                    pred = torch.from_numpy(pred_np)

                elif model_name == "VARLINGAM":
                    pred_np = run_varlingam_with_bootstrap(
                        sample=X,
                        max_lag=max_lag,
                        n_sampling=N_SAMPLING,
                        min_causal_effect=0.05,
                    )
                    pred = torch.from_numpy(pred_np)

                else:
                    raise NotImplementedError(f"Unknown model type: {model_name}")

                tac = time.time()
                running_times_dict[model_name].append(round((tac - tic) / N_SAMPLING, 3))

                if Y.sum() <= 0 or Y.sum() == np.prod(Y.shape):
                    continue
                if pred.sum() <= 0 or pred.sum() == np.prod(pred.shape):
                    continue

                Y[Y < 0.05] = 0
                Y[Y >= 0.05] = 1

                tpr, fpr, tnr, fnr, auc = custom_binary_metrics(pred, Y, verbose=False)

                precision = tpr / (tpr + fpr) if tpr + fpr != 0 else 0
                recall = tpr / (tpr + fnr) if tpr + fnr != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

                tpr_list.append(float(tpr))
                fpr_list.append(float(fpr))
                tnr_list.append(float(tnr))
                fnr_list.append(float(fnr))
                auc_list.append(float(auc))
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

                per_sample_results[model_name].append((filename, auc))

            results_dict[model_name] = [token for token in auc_list]
            for k, v in zip(["AUC","TPR","FPR","TNR","FNR","Precision","Recall","F1"],
                            [auc_list,tpr_list,fpr_list,tnr_list,fnr_list,precision_list,recall_list,f1_list]):
                run_metrics[k].append(v)   # full list, no mean yet

        def mean_std_allruns(x):
            # Flatten all scores across seeds
            flat = np.array([s for run in x for s in run], dtype=float)
            mu = flat.mean()
            std = flat.std()
            return mu, std

        means_ses = {k: mean_std_allruns(v) for k, v in run_metrics.items()} # NOTE: Standard errors are computed across all datasets and seeds, while in the thesis across seeds only

        results_df.loc[len(results_df), :] = [
            model_name,
            means_ses["AUC"][0], means_ses["AUC"][1],
            means_ses["TPR"][0], means_ses["TPR"][1],
            means_ses["FPR"][0], means_ses["FPR"][1],
            means_ses["TNR"][0], means_ses["TNR"][1],
            means_ses["FNR"][0], means_ses["FNR"][1],
            means_ses["Precision"][0], means_ses["Precision"][1],
            means_ses["Recall"][0], means_ses["Recall"][1],
            means_ses["F1"][0], means_ses["F1"][1],
        ]

    display(results_df)

    save_path = out_dir / "cdml_results_metrics.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\nMetrics saved to: {save_path}")

    model_pairs = list(combinations(per_sample_results.keys(), 2))
    if len(model_pairs) > 0:
        pairwise_df = perform_wilcoxon_test(per_sample_results)
        display(pairwise_df)
        #pairwise_df.to_csv(out_dir / "pairwise_significance_cdml.csv", index=False)

    running_times_df = pd.DataFrame({k: pd.Series(v) for k, v in running_times_dict.items()})
    running_times_summary = running_times_df.aggregate(['mean','median','std','min','max']).T
    running_times_summary.to_csv(out_dir / "running_times_summary_cdml.csv", index=True)

    print(f"Running time summary saved to: {out_dir / 'running_times_summary_cdml.csv'}")

    # save per-sample results dict to .npy
    np.save(out_dir / f"per_sample_results_cdml.npy", per_sample_results)

    critical_diff_df = pd.DataFrame(columns=["model", "dataset", "accuracy"])

    for model, auc_list in results_dict.items():
        for auc in auc_list:
                critical_diff_df.loc[len(critical_diff_df)] = [model, 'CDML', auc]

    critical_diff_df.to_csv(out_dir / f"critical_diff_df_CDML.csv", index=False)




def optimal_threshold_youden(y_true: np.ndarray, y_score: np.ndarray, bin_thresh: float=0.05) -> tuple:
    """
    Compute optimal threshold using Youden's J statistic = TPR - FPR.
    Handles continuous y_true by binarizing with bin_thresh.
    """
    y_true_flat = y_true.flatten()
    y_score_flat = y_score.flatten()

    # Binarize y_true for ROC
    y_true_bin = (y_true_flat >= bin_thresh).astype(int)

    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score_flat)
    j_scores = tpr - fpr
    j_best = j_scores.argmax()

    return thresholds[j_best], fpr, tpr, thresholds


def threshold_by_density(pred: torch.Tensor, target_density: float) -> tuple:
    """
    Threshold continuous adjacency tensor by matching target density.
    """
    flat_pred = pred.flatten()
    k = int(len(flat_pred) * target_density)
    if k <= 0:
        return torch.zeros_like(pred), 0.0
    thresh = torch.topk(flat_pred, k).values.min().item()
    binarized = (pred >= thresh).float()

    return binarized, thresh

def threshold_by_auc(y_true: np.ndarray, y_score: np.ndarray, bin_thresh=0.05) -> float:
    """
    Determines the optimal threshold for a set of predicted scores (y_score) that maximizes the Area Under the Curve (AUC)
    for binary classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels or continuous values. Binarized using the `bin_thresh` parameter.
        y_score (np.ndarray): Predicted scores or probabilities for the positive class.
        bin_thresh (float, optional): Threshold for binarizing `y_true` into binary labels. Defaults to 0.05.

    Returns:
        float: The threshold value for `y_score` that yields the highest AUC.

    Notes:
        - The function iterates over a range of thresholds (from 0 to 1 with a step of 0.01) to compute the AUC for 
          each threshold.
        - The `custom_binary_metrics` function is used to calculate the AUC for the given predictions and ground truth.
        - The best threshold and corresponding AUC are printed to the console.
    """
    y_true_flat = y_true.flatten()
    y_true = (y_true_flat >= bin_thresh).astype(int)
    thresholds = np.arange(0, 1, 0.01)

    best_auc = 0
    best_thresh = 0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        _, _, _, _, auc = custom_binary_metrics(y_pred, A=y_true, verbose=False)
        if auc > best_auc:
            best_auc = auc
            best_thresh = t

    print(f"best threshold: {best_thresh}, best AUC: {best_auc}")

    return best_thresh

def right_shift(arr: np.ndarray, shift_by: int=1) -> np.ndarray: # function to create time lagged causal relationships
    """
    Shifts a numpy array to the right by a specified number of positions.

    Args:
        arr (np.ndarray): The input array to be shifted.
        shift_by (int): The number of positions to shift the array to the right.

    Returns: 
        np.ndarray: The shifted array.
    """
    arr = list(arr)
    shift_by = shift_by % len(arr)  

    return np.array(arr[-shift_by:] + arr[:-shift_by])


def run_illustrative_example(n: int) -> tuple:
    """
    Creates a synthetic example to show the input data structure. Each time series corresponds to a different column in the DataFrame.
    The example consists of 3 variables V_1, V_2, and V_3, with the following causal relationships:
        V_1 -> V_2 with lag 1
        V_1 -> V_3 with lag 3
        V_2 -> V_3 with lag 2
    
    The temporal SCM is of the form $V_1(t) = \epsilon(t), V_2(t) = 3 * V_1(t-1) + \epsilon(t), V_3(t) = V_2(t-2) + 5 * V_1(t-3) + \epsilon(t)$
    where $\epsilon(t)$ is Gaussian noise.

    Args:
        n (int): The number of time steps to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic data.
        torch.Tensor: The ground truth lagged adjacency graph. 
    """
    MAX_LAG = 3
    V1 = np.random.normal(size=n, loc = 0, scale = 1)
    V2 = 3 * right_shift(V1, shift_by = 1) + np.random.normal(size=n, loc = 0, scale = 1)  # B(t) = 0.5 * A(t-1) + noise
    V3 = right_shift(V2, shift_by = 2) + 5 * right_shift(V1, shift_by = 3) + np.random.normal(size=n, loc = 0, scale = 1)  # C(t) = 0.6 * B(t-2) + noise
    
    df = pd.DataFrame({'V_1': V1, 'V_2': V2, 'V_3': V3})

    data = torch.tensor(df.values, dtype=torch.float32)

    # creating true lagged adj. tensor
    Y_cpd = torch.zeros((data.shape[1], data.shape[1], MAX_LAG))
    Y_cpd[1, 0, 2] = 1 # A -> B with lag 1, so last dim is \ell_max - 1 = 2
    Y_cpd[2, 0, 0] = 1 # A -> C with lag 3, so last dim is \ell_max - 3 = 0
    Y_cpd[2, 1, 1] = 1 # B -> C with lag 2, so last dim is \ell_max - 2 = 1 

    return df, Y_cpd