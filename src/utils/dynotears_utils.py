import string

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.utils.causalnex.structure.dynotears import from_pandas_dynamic
#from causalnex.structure import structure_model

from src.utils.transformation_utils import (_from_full_to_lagged_adj,
                                            group_lagged_nodes)

""" _____________________________________________ DYNOTEARS _____________________________________________ """

def reverse_order_sm(sm) -> list:
    """
    Returns the reversed order of the nodes of the SM object, similar to the SCM generator.
    See the custom generator for details.

    Args:
        sm (StructureModel): the structure predicted by dynotears
    
    Returns:
        list: containing the reversed order of nodes
    """
    n_lags = max([int(node.split("lag")[-1]) for node in sm.nodes if ("lag" in node)])
    temp = [reversed(sorted([col for col in sm.nodes if col.endswith(f"lag{t}")])) for t in reversed(range(n_lags + 1))]
    temp = [x for y in temp for x in y]

    return temp


def regular_order_sm(sm) -> list:
    """
    Returns the reversed order of the nodes of the SM object, similar to the SCM generator.
    See the custom generator for details.

    Args:de
        sm (StructureModel): the structure predicted by dynotears

    Returns:
        (list) containing the reversed order of nodes
    """
    n_lags = max([int(node.split("lag")[-1]) for node in sm.nodes if ("lag" in node)])
    temp = [sorted([col for col in sm.nodes if col.endswith(f"lag{t}")]) for t in range(n_lags + 1)]
    temp = [x for y in temp for x in y]

    return temp


def rename_sm_nodes(pred_pd: pd.DataFrame) -> pd.DataFrame:
    """ 
    Renames the nodes of the Pandas adjacency matrix of SM accordingly, to achieve compatibility with the existing functions

    Args:
        pred_pd (pd.DataFrame): the adjacency matrix of the DYNOTEARS output 
    
    Returns:
        pred_pd (pd.DataFrame): the Pandas adjacency matrix, renamed
    """
    #pred_pd = pred_pd.rename(
    #    columns=dict(zip(
    #        [col for col in pred_pd.columns],
    #        [col.replace('_lag', '_t-') for col in pred_pd.columns] 
    #    )),
    #    index=dict(zip(
    #        [col for col in pred_pd.columns],
    #        [col.replace('_lag', '_t-') for col in pred_pd.columns] 
    #    ))
    #)
    #pred_pd = pred_pd.rename(
    #    columns=dict(zip(
    #        [col for col in pred_pd.columns],
    #        [col.replace('_t-0', '_t') for col in pred_pd.columns] 
    #    )), 
    #    index=dict(zip(
    #        [col for col in pred_pd.columns],
    #        [col.replace('_t-0', '_t') for col in pred_pd.columns] 
    #    )), 
    #)
    pred_pd = pred_pd.rename(
        columns={col: col.replace('_lag', '_t-') for col in pred_pd.columns},
        index={idx: idx.replace('_lag', '_t-') for idx in pred_pd.index}
    )
    pred_pd = pred_pd.rename(
        columns={col: col.replace('_t-0', '_t') for col in pred_pd.columns},
        index={idx: idx.replace('_t-0', '_t') for idx in pred_pd.index}
    )

    return pred_pd


def run_dynotears(data: pd.DataFrame, n_lags: int, lambda_w: float=0.1, lambda_a: float=0.1) -> pd.DataFrame:
    """ 
    Runs the DYNOTEARS algorithm given a time-series dataset as a Pandas DataFrame. 

    Args:
        data (pd.DataFrame): the input time-series
        n_lags (int): the maximum number of lags
        lambda_w (float): the lambda_w internal parameter of DYNOTEARS; default value is `0.1`
        lambda_a (float): the lambda_a internal parameter of DYNOTEARS; default value is `0.1`

    Returns (pd.DataFrame):
        the full time adjacency matrix as a pd.DataFrame 
    """

    # Rename columns to avoid duplicates when the initial names are numbers
    data.rename(columns=dict(zip(data.columns, list(string.ascii_uppercase)[:len(list(data.columns))])), inplace=True)

    sm = from_pandas_dynamic(
        time_series = data,
        p = n_lags,
        lambda_w = lambda_w,
        lambda_a = lambda_a,
        w_threshold=0.05,
        max_iter=100,
    )

    # convert to Pandas adjacency
    pred_pd = nx.to_pandas_adjacency(sm, nodelist=reverse_order_sm(sm))
    
    # rename nodes
    pred_pd = rename_sm_nodes(pred_pd=pred_pd)

    # Remove contemporaneous nodes from prediction
    pred_pd.loc[group_lagged_nodes(pred_pd.columns)['0'], group_lagged_nodes(pred_pd.columns)['0']] = \
        np.zeros(shape=pred_pd.loc[group_lagged_nodes(pred_pd.columns)['0'], group_lagged_nodes(pred_pd.columns)['0']].shape)

    return pred_pd

def dynotears_to_tensor(W: np.ndarray, n_vars: int, max_lag: int) -> np.ndarray:
    """
    Converts the output of the DYNOTEARS algorithm to a tensor of shape (n_vars, n_vars, max_lag) that supports our convention 
    (i.e. the first dimension is the source node, the second is the target node, and the third is max_lag-lag)

    Args:
        W (np.ndarray): the output of the DYNOTEARS algorithm
        n_vars (int): the number of variables in the time-series
        max_lag (int): the maximum number of lags

    Returns:
        adj (np.ndarray): the tensor of shape (n_vars, n_vars, max_lag)
    
    """

    adj = np.zeros((n_vars, n_vars, max_lag))
    for i in range(n_vars):
        for j in range(n_vars):
            for lag in range(1, max_lag+1):
                idx = (lag-1)*n_vars + j
                val = W[idx, i]
                adj[i, j, max_lag-lag] = val

    return adj

def run_dynotears_with_bootstrap(ts: pd.DataFrame, n_lags: int, n_bootstrap: int = 10, lambda_w: float = 0.1, lambda_a: float = 0.1) -> np.ndarray:
    """
    Runs the DYNOTEARS algorithm with bootstrapping given a time-series dataset as a Pandas DataFrame.
    As DYNOTEARS returns causal effects instead of edge confidence, we run bootstrapping to obtain them.

    Args:
        ts (pd.DataFrame): the input time-series
        n_lags (int): the maximum number of lags
        n_bootstrap (int): the number of bootstrap samples; default value is `10`
        lambda_w (float): the lambda_w internal parameter of DYNOTEARS; default value is `0.1`
        lambda_a (float): the lambda_a internal parameter of DYNOTEARS; default value is `0.1`

    Returns (np.ndarray):
        the full time adjacency matrix as a np.ndarray
    """

    n_samples, n_vars = ts.shape[0], ts.shape[1]
    lagged_sum = np.zeros((n_vars, n_vars, n_lags))

    ts_np = ts.to_numpy() if hasattr(ts, "to_numpy") else ts
    for b in range(n_bootstrap):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        ts_boot = ts_np[idxs]
        pred_pd = run_dynotears(
            data=pd.DataFrame(ts_boot),
            n_lags=n_lags,
            lambda_w=lambda_w,
            lambda_a=lambda_a
        )
        pred = _from_full_to_lagged_adj(pred_pd)
        if hasattr(pred, "numpy"):
            pred = pred.numpy()
        lagged_sum += np.abs(pred)   # absolute causal edge confidence 

    mean_abs_weight = lagged_sum / n_bootstrap

    return mean_abs_weight / mean_abs_weight.max() # normalize