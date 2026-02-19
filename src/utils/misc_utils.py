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


def get_aligned_aucs(results_a, results_b):
    """
    Align per-sample AUCs from two models.

    Expects each input to be an iterable of (sample_id, auc).
    results_* = list of (sample_id, auc)
    """

    dict_a = dict(results_a)
    dict_b = dict(results_b)

    common_ids = sorted(set(dict_a.keys()) & set(dict_b.keys()))

    scores_a = [dict_a[i] for i in common_ids]
    scores_b = [dict_b[i] for i in common_ids]

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

def summarize_against_reference_model(pairwise_df, reference_model):
    rows = pairwise_df[
        (pairwise_df["model_a"] == reference_model) |
        (pairwise_df["model_b"] == reference_model)
    ].copy()

    def normalize(row):
        if row["model_a"] == reference_model:
            return {
                "model": row["model_b"],
                "mean_b": row["mean_b"],
                "std_b": row["std_b"],
                "median_delta": -row["median_delta"],
                "mean_delta": -row["mean_delta"],
                "raw_p_value": row["raw_p_value"],
                "adjusted_alpha": row["adjusted_alpha"],
                "significant": row["significant_after_correction"],
            }
        else:
            return {
                "model": row["model_a"],
                "mean_a": row["mean_a"],
                "std_a": row["std_a"],
                "median_delta": row["median_delta"],
                "mean_delta": row["mean_delta"],
                "raw_p_value": row["raw_p_value"],
                "adjusted_alpha": row["adjusted_alpha"],
                "significant": row["significant_after_correction"],
            }

    return pd.DataFrame(rows.apply(normalize, axis=1).tolist())


def bootstrap_paired_delta(
    delta: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95,
    statistic: str = "mean",
    seed: int = 0,
):
    """
    Paired bootstrap confidence interval for delta = scores_a - scores_b.

    Args:
        delta: 1D array of paired differences
        n_boot: number of bootstrap resamples
        ci: confidence level (e.g., 95)
        statistic: "mean" or "median"
        seed: random seed

    Returns:
        (center, ci_low, ci_high)
    """
    assert statistic in {"mean", "median"}

    rng = np.random.default_rng(seed)
    n = len(delta)

    if n < 5:
        return np.nan, np.nan, np.nan

    idx = rng.integers(0, n, size=(n_boot, n))
    samples = delta[idx]

    if statistic == "mean":
        stats = samples.mean(axis=1)
        center = delta.mean()
    else:
        stats = np.median(samples, axis=1)
        center = np.median(delta)

    alpha = (100 - ci) / 2
    lo = np.percentile(stats, alpha)
    hi = np.percentile(stats, 100 - alpha)

    return center, lo, hi


def perform_wilcoxon_test(per_sample_results: dict, metric: str = "AUC",
                          adjust_for_multiple_tests: bool = True, alpha: float = 0.05) -> pd.DataFrame:
    """
    Performs the paired Wilcoxon signed-rank test on aligned per-sample metrics.
    Optionally computes bootstrap confidence intervals for paired deltas.

    per_sample_results:
        dict[model_name] -> list of ((shard_idx, sample_idx), auc)
    """
    model_pairs = list(combinations(per_sample_results.keys(), 2))
    if not model_pairs:
        raise ValueError("No model pairs found.")

    corrected_alpha = alpha / len(model_pairs) if adjust_for_multiple_tests else alpha

    rows = []
    for model_a, model_b in model_pairs:

        dict_a = dict(per_sample_results[model_a])
        dict_b = dict(per_sample_results[model_b])

        common_ids = sorted(set(dict_a.keys()) & set(dict_b.keys()))

        if len(common_ids) == 0:
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_shared": 0,
                "mean_a": np.nan,
                "mean_b": np.nan,
                "mean_delta": np.nan,
                "p_value": np.nan,
                "significant": False,
            })
            continue

        scores_a = np.array([dict_a[i] for i in common_ids], dtype=float)
        scores_b = np.array([dict_b[i] for i in common_ids], dtype=float)

        mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
        scores_a = scores_a[mask]
        scores_b = scores_b[mask]

        delta = scores_a - scores_b
        n = len(delta)

        if n < 5:
            p_val = np.nan
        else:
            nonzero_mask = delta != 0
            scores_a_nz = scores_a[nonzero_mask]
            scores_b_nz = scores_b[nonzero_mask]

            if len(scores_a_nz) < 5:
                p_val = 1.0
            else:
                try:
                    _, p_val = wilcoxon(
                        scores_a_nz,
                        scores_b_nz,
                        alternative="two-sided",
                        zero_method="wilcox"
                    )
                except Exception:
                    p_val = np.nan

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "mean_a": scores_a.mean() if n > 0 else np.nan,
            "mean_b": scores_b.mean() if n > 0 else np.nan,
            "std_a": scores_a.std(ddof=1) if n > 1 else np.nan,
            "std_b": scores_b.std(ddof=1) if n > 1 else np.nan,
            "mean_delta": delta.mean() if n > 0 else np.nan,
            "median_delta": np.median(delta) if n > 0 else np.nan,
            "raw_p_value": p_val,
            "adjusted_alpha": corrected_alpha,
            "significant": (
                p_val < corrected_alpha if not np.isnan(p_val) else False
            ),
        })

    return pd.DataFrame(rows)


def load_full_dataset(base_path: Path, split: str):
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

    if not isinstance(data, list):
        raise ValueError(f"{file_path} does not contain a list of samples")

    print(f"Loaded dataset file: {file_path} ({len(data)} samples)")

    return data  # returns list of (X,Y)


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
        shard_files = sorted(split_path.glob(f"{split}_shard*.pt"))

    if not shard_files:
        single_file = split_path / f"{split}.pt"
        if single_file.exists():
            return [torch.load(single_file, weights_only=False)]
        else:
            raise FileNotFoundError(f"No dataset found for split {split}")

    print(f"- Found {len(shard_files)} shard(s)")

    all_shards = []

    for shard_path in shard_files:
        shard_data = torch.load(shard_path, weights_only=False)

        if not isinstance(shard_data, list):
            raise ValueError(f"{shard_path} is not a list of samples")

        print(f"├── Loaded {shard_path.name} ({len(shard_data)} samples)")
        all_shards.append(shard_data)

    return all_shards


def load_fmri_datasets(fmri_path: Path, verbose: bool = False):
    timeseries_files = sorted([f for f in os.listdir(fmri_path) if 'timeseries' in f])
    ground_truth_files = sorted([f for f in os.listdir(fmri_path) if 'gt_processed' in f])

    dataset = [[]]  # single shard

    for ts_file in timeseries_files:
        ts_number = extract_number(ts_file, r'timeseries(\d+)\.csv')

        for gt_file in ground_truth_files:
            gt_number = extract_number(gt_file, r'sim(\d+)_gt_processed\.csv')

            if ts_number == gt_number:

                X_raw = pd.read_csv(fmri_path / ts_file)
                Y_raw = pd.read_csv(fmri_path / gt_file, names=['effect','cause','delay'])

                X = torch.tensor(X_raw.values, dtype=torch.float32)
                Y = from_fmri_to_lagged_adj(X_raw, Y_raw)

                if Y.sum() <= 0 or Y.sum() == np.prod(Y.shape):
                    continue

                dataset[0].append((X, Y))

    if verbose:
        print(f"Loaded {len(dataset[0])} fMRI samples")

    return dataset


def load_cdml_datasets(cdml_path: Path, verbose: bool = False, MAX_VAR: int = 12, MAX_LAG: int = 3):

    filenames = sorted(f.split("_data.csv")[0] for f in os.listdir(cdml_path) if f.endswith("_data.csv"))

    dataset = [[]]  # single shard
    for fname in filenames:

        X_raw = pd.read_csv(cdml_path / f"{fname}_data.csv")
        Y_raw = pd.read_csv(cdml_path / f"{fname}_target.csv", index_col="Unnamed: 0")

        X = torch.tensor(X_raw.values, dtype=torch.float32)
        X = (X - X.min()) / (X.max() - X.min() + 1e-8)

        Y = y_from_cdml_to_lagged_adj(Y_raw)

        if X.shape[1] > MAX_VAR:
            continue
        if Y.shape[1] > MAX_VAR or Y.shape[2] > MAX_LAG:
            continue
        if Y.sum() <= 0 or Y.sum() == np.prod(Y.shape):
            continue

        dataset[0].append((X, Y))

    if verbose:
        print(f"Loaded {len(dataset[0])} CDML samples")

    return dataset


def load_dataset(
    cpd_path: Path = None,
    split: str = "test",
    sharded_data: bool = False,
    fmri_data: bool = False,
    cdml_path: Path = None,
):
    """
    Returns:
        dataset: List[List[(X, Y)]]
                 Outer list = shards
                 Inner list = samples
    """

    if cdml_path is not None:
        return load_cdml_datasets(Path(cdml_path))

    if fmri_data:
        return load_fmri_datasets(Path(cpd_path))

    if sharded_data:
        return load_sharded_dataset(Path(cpd_path), split)

    return [load_full_dataset(Path(cpd_path), split)]


def plot_running_times(
    running_times_dict,
    save_dir,
    dataset_label,
    output_format="pdf"  # use pdf for paper
):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    labels = list(running_times_dict.keys())
    values = list(running_times_dict.values())

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


    # Create boxplot with controlled styling
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )
    SOFT_COLORS = [
        "#DB81D7", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
        "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"
    ]

    # Apply soft colors
    for patch, color in zip(bp["boxes"], SOFT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Subtle grid like your other plots
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Clean spines (paper style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Labels and title
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Avg running time per dataset (sec)", fontsize=12)
    ax.set_title("Distribution of Running Times", fontsize=14)

    plt.tight_layout()

    output_path = save_dir / f"running_times_{dataset_label}.{output_format}"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


def run_evaluation_experiments(models: dict,
                               cpd_path: Path = None,
                               out_dir: Path = None,
                               split: str = "test",
                               sharded_data: bool = False,
                               fmri_data: bool = False,
                               kuramoto_data: bool = False,
                               cdml_path: Path = None,
                               dataset_label: str = None,
                               N_SAMPLING: int = 10,
                               N_RUNS: int = 5,
                               MAX_VAR: int = 12,
                               MAX_LAG: int = 3) -> None:
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
    if cdml_path is not None:
        data_path = Path(cdml_path)
    else:
        data_path = Path(cpd_path)

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

    print(f'Dataset: {data_path.parent.stem}')

    """ Load data depending on mode """
    if cdml_path is not None:
        cdml_path = Path(cdml_path)
        dataset_iterator = load_cdml_datasets(cdml_path, verbose=True)

    elif fmri_data:
        fmri_path = Path(cpd_path)
        dataset_iterator = load_fmri_datasets(fmri_path, verbose=True)

    else:
        if sharded_data:
            print("--SHARDED mode--")
            dataset_iterator = load_sharded_dataset(cpd_path, split)
        else:
            print("--SINGLE FILE mode--")
            dataset_iterator = [load_full_dataset(cpd_path, split)]

    """ Model loop """
    for model_name, model in zip(models.keys(), models.values()):

        print(f"\n___{model_name}___")

        MAX_VAR_ = MAX_VAR
        MAX_LAG_ = MAX_LAG
        if model_name == "CP_trf":
            MAX_VAR_ = 5
        elif "LCM" in model_name:
            MAX_VAR_, MAX_LAG_ = 12, 3

        if kuramoto_data:
            if ("PCMCI" in model_name) or ("DYNOTEARS" in model_name) or ("VARLINGAM" in model_name):
                if "kuramoto_5" in str(cpd_path):
                    MAX_VAR_ = 5
                    MAX_LAG_ = 1 
                else:
                    MAX_VAR_ = 10
                    MAX_LAG_ = 1 

        print(f"VAR: {MAX_VAR_} | MAX LAG: {MAX_LAG_}") if "LCM" or "CP_trf" in model_name else print(f"MAX LAG: {MAX_LAG_}") 

        # store metrics across multiple runs
        run_metrics = {"AUC": [], "TPR": [], "FPR": [], "TNR": [], "FNR": [], "Precision": [], "Recall": [], "F1": []}

        # PCMCI is deterministic, multiple runs are not necessary
        if model_name in ["PCMCI", "DYNOTEARS", "VARLINGAM"]:
            N_RUNS = 1

        for run_id in range(N_RUNS):
            seed = 42 + run_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            print(f"\n |--- Run {run_id+1}/{N_RUNS} (seed={seed}) ---")

            tpr_list, fpr_list, tnr_list, fnr_list, auc_list = [], [], [], [], []
            precision_list, recall_list, f1_list = [], [], []

            for shard_idx, data in enumerate(dataset_iterator):
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

                            #tpr, fpr, tnr, fnr, auc = custom_binary_metrics(binary=pred_bin, A=Y_cpd, verbose=False)
                            tpr, fpr, tnr, fnr, auc = custom_binary_metrics(pred=pred, A=Y_cpd, verbose=False)

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

                            #tpr, fpr, tnr, fnr, auc = custom_binary_metrics(torch.tensor(binary_adj_fixed), Y_cpd, verbose=False)
                            tpr, fpr, tnr, fnr, auc = custom_binary_metrics(torch.tensor(score_adj), Y_cpd, verbose=False)

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
                            assert X_cpd.shape[1]<=MAX_VAR_, \
                                f"ValueError: input time-series have {X_cpd.shape[1]} variables, while the current model supports at most {MAX_VAR_}"
                            assert Y_cpd.shape[1]<=MAX_VAR_, \
                                f"ValueError: input adjacency matrix has {Y_cpd.shape[1]} nodes, while the current model supports at most {MAX_VAR_}"
                            assert Y_cpd.shape[2]<=MAX_LAG_, \
                                f"ValueError: input adjacency matrix has {Y_cpd.shape[2]}, while the current model supports at most {MAX_LAG_}"

                            # Padding
                            VAR_DIF = MAX_VAR_ - X_cpd.shape[1]
                            LAG_DIF = MAX_LAG_ -Y_cpd.shape[2]
                            
                            if VAR_DIF > 0: # if the number of variables is less than the maximum
                                X_cpd = torch.concat(
                                    [X_cpd,torch.normal(0, 0.01, (X_cpd.shape[0], VAR_DIF))], axis=1 # pad with noise
                                )
                                Y_cpd = torch.nn.functional.pad(
                                    Y_cpd, (0, 0, 0, VAR_DIF, 0, VAR_DIF), mode="constant", value=0.0 # pad the adjacency matrix with zeros
                                )
                            if LAG_DIF > 0: # if the number of lags is less than the maximum
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
                            tpr, fpr, tnr, fnr, auc = custom_binary_metrics(pred=pred[0], A=Y_cpd, verbose=False)

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

                        #per_sample_results[model_name].append((shard_idx, auc))
                        sample_id = (shard_idx, idx)
                        per_sample_results[model_name].append((sample_id, float(auc)))

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

        # NOTE: Standard errors are computed across all datasets and seeds, while in the thesis across each seed only
        means_ses = {k: mean_std_allruns(v) for k, v in run_metrics.items()}

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
        dataset_label = data_path.stem
    save_dir = out_dir / dataset_label
    save_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(save_dir / f"results_metrics_{dataset_label}.csv", index=False)
    print(f"\nMetrics saved to: {save_dir}")

    model_pairs = list(combinations(per_sample_results.keys(), 2))
    if len(model_pairs) > 0:
        pairwise_df = perform_wilcoxon_test(per_sample_results)
    else:
        raise ValueError("No model pairs found.") 

    pairwise_df = perform_wilcoxon_test(per_sample_results) 

    display(pairwise_df)
    pairwise_df.to_csv(save_dir / f"pairwise_significance_{dataset_label}.csv", index=False)

    plot_running_times(
        running_times_dict,
        save_dir,
        dataset_label,
        output_format="pdf"
    )
    print(f"Figure saved to: {out_dir} / running_times_{dataset_label}.pdf")

    running_times_df = pd.DataFrame({k: pd.Series(v) for k, v in running_times_dict.items()})
    running_times_summary = running_times_df.aggregate(['mean', 'median', 'std', 'min', 'max']).T
    running_times_summary.index = [
        f"{label} (running time in sec)" for label in running_times_summary.index
    ]
    running_times_summary.to_csv(save_dir / f"running_times_summary_{dataset_label}.csv", index=False)

    # save per-sample results dict to .npy
    #np.save(save_dir / f"per_sample_results_{dataset_label}.npy", per_sample_results)

    # if save_critical_diff:
        #critical_diff_df = pd.DataFrame(columns=["model", "dataset", "accuracy"])
        #for model, auc_list in results_dict.items():
        #    for auc in auc_list:
        #            critical_diff_df.loc[len(critical_diff_df)] = [model, 'CDML', auc]
        #critical_diff_df.to_csv(out_dir / f"critical_diff_df_CDML.csv", index=False)

    print(f"\n Results saved to {out_dir}")


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