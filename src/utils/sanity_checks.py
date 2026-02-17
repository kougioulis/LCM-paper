from typing import Union

import numpy as np
import pandas as pd
import torch

from utils.transformation_utils import regular_order_pd


def print_sum_of_edges(adj_pd: pd.DataFrame, adj_lagged: Union[torch.Tensor, np.ndarray], assertive: bool=False) -> None:
    """ 
    Print and compare the total number of edges in both representations. The effects are only considered by the current time-step. 
    They should always amount to the same number for each node, in both cases. Assumes that the order of the nodes in the 
    full-time adjacency matrix is ascending based on the time lag and alphabetical in the time-lag level. Method should work with both 
    the full-time adjacency matrix and the part-time adjacency matrix, where the edges showing the causal stationarity are missing.  

    Args:
        adj_pd (pd.DataFrame) : the full-time adjacency matrix
        adj_lagged (torch.Tensor or numpy.array) : the lagged adjacency matrix
        assertive (bool) : if True and the sums are not equal, it throws an Assert error

    Returns:
        None
    """
    regular_order_nodes = regular_order_pd(adj_pd=adj_pd)
    adj_pd = adj_pd.loc[regular_order_nodes, regular_order_nodes]
    lagged_sum = adj_lagged.sum(dtype=int).item()
    pd_sum = adj_pd.loc[:, regular_order_nodes[:adj_lagged.shape[1]]].to_numpy().sum()

    if assertive:
        assert lagged_sum==pd_sum, f"ValueError: the edge sum of the lagged representation ({lagged_sum}) \
            is not equal to the edge sum of the PD representation ({pd_sum})"
    print(f"(lagged) {lagged_sum} - vs - {pd_sum} (pd) \n")


def print_sum_of_causes(adj_pd: pd.DataFrame, adj_lagged: Union[torch.Tensor, np.ndarray], assertive: bool=False) -> None:
    """ 
    Print and compare the total number of causes for each effect. The effects are only considered by the current time-step. 
    They should always amount to the same number for each node, in both representations. Assumes that the order of the nodes in the 
    full-time adjacency matrix is ascending based on the time lag and alphabetical in the time-lag level. Method should work with both 
    the full-time adjacency matrix and the part-time adjacency matrix, where the edges showing the causal stationarity are missing.  

    Args:
        adj_lagged (torch.Tensor or numpy.array) : the lagged adjacency matrix
        adj_pd (pd.DataFrame) : the full-time adjacency matrix
        assertive (bool) : if True and the sums are not equal, it throws an Assert error

    Returns:
        None
    """
    regular_order_nodes = regular_order_pd(adj_pd=adj_pd)
    adj_pd = adj_pd.loc[regular_order_nodes, regular_order_nodes] 

    for idx, col in enumerate(list(adj_pd.columns)[:adj_lagged.shape[1]]):
        lagged_causes = adj_lagged[idx, :, :].sum().int().item()
        pd_causes = adj_pd[col].sum()
        if assertive:
            assert lagged_causes==pd_causes, f"ValueError: the number of causes for node {col} of the lagged representation ({lagged_causes}) \
                is not equal to the number of causes of the PD representation ({pd_causes})"
        print(f"- {col} -> (lagged) {lagged_causes} vs {pd_causes} (full) \n")


def print_sum_of_causes_fmri(edgelist: pd.DataFrame, adj_pd: pd.DataFrame, adj_lagged: Union[torch.Tensor, np.ndarray], assertive: bool=False) -> None:
    """ 
    Print and compare the total number of causes for each effect. The effects are only considered by the current time-step. 
    They should always amount to the same number for each node, in both representations. Assumes that the order of the nodes in the 
    full-time adjacency matrix is ascending based on the time lag and alphabetical in the time-lag level. Method should work with both 
    the full-time adjacency matrix and the part-time adjacency matrix, where the edges showing the causal stationarity are missing.  

    Args:
        edgelist (pd.DataFrame) : the true fMRI edgelist
        adj_lagged (torch.Tensor or numpy.array) : the lagged adjacency matrix
        adj_pd (pd.DataFrame) : the full-time adjacency matrix
        assertive (bool) : if True and the sums are not equal, it throws an Assert error

    Returns:
        None
    """
    print("Edgelist vs Lagged")
    for idx, node in enumerate(sorted(edgelist['effect'].unique())):
        lagged_causes = adj_lagged[idx, :, :].sum().int().item()
        # pd_causes = adj_pd[col].sum()
        edgelist_causes = edgelist[edgelist['effect']==node].shape[0]

        if assertive:
            assert lagged_causes==edgelist_causes, f"ValueError: the number of causes for node {node} of the lagged representation ({lagged_causes}) \
                is not equal to the number of causes of the PD representation ({edgelist_causes})"

        print(f"- {node}: (edgelist) {edgelist_causes} vs {lagged_causes} (lagged) \n")

    print("Edgelist vs PD")
    regular_order_nodes = regular_order_pd(adj_pd=adj_pd)
    adj_pd = adj_pd.loc[regular_order_nodes, regular_order_nodes] 
    for idx, node in enumerate(list(adj_pd.columns)[:len(edgelist['effect'].unique())]):
        # display(adj_pd[node])
        pd_causes = adj_pd[node].sum()
        edgelist_causes = edgelist[edgelist['effect']==idx].shape[0]

        if assertive:
            assert lagged_causes==edgelist_causes, f"ValueError: the number of causes for node {node} of the lagged representation ({pd_causes}) \
                is not equal tothe number of causes of the PD representation ({edgelist_causes})"

        print(f"- {node}: (edgelist) {edgelist_causes} vs {pd_causes} (pd)")
    print()