import string

import networkx as nx
import numpy as np
import pandas as pd
import torch

""" _____________________________________________ Generic _____________________________________________ """

def to_lagged_adj_ready(lagged_adj):
    """ 
    Transpose each time slice, to match the lagged adjacency notation.

    Args:
        lagged_adj (numpy.array or torch.Tensor): the lagged adjacency matrix
    
    Returns:
        the inversed lagged adjacency matrix
    """
    if not isinstance(lagged_adj, torch.Tensor):
        lagged_adj = torch.tensor(lagged_adj)
    structure_T = torch.zeros_like(lagged_adj)
    for t in range(structure_T.shape[2]):
        structure_T[:, :, t] = lagged_adj[:, :, t].T

    return structure_T


def group_lagged_nodes(lagged_nodes) -> dict:
    """ 
    Returns a dictionary with the lags as str and the corresponding sublist of lagged nodes

    Args:
        lagged_nodes (str): the lagged nodes

    Returns:
        a dictionary with the lags as str and the corresponding sublist of lagged nodes
    """
    # get the number of lags
    n_lags = max([int(node.split("_t-")[-1]) for node in lagged_nodes if ("_t-" in node)])

    # create the dictionary
    lag_dict = {}
    for t in range(n_lags + 1):
        if t==0:
            lag_dict[f"{t}"] = [node for node in lagged_nodes if ("_t" in node) and ("_t-" not in node)]
        else:
            lag_dict[f"{t}"] = [node for node in lagged_nodes if (f"_t-{t}" in node)]

    return lag_dict


def reverse_order_pd(adj_pd) -> list:
    """
    Returns the reversed order of the nodes of the Pandas full-time adjacency matrix, similar to the custom generator. 
    See the custom generator for details. Assumes the dataframe follows the node naming convention based on '_t-'.

    Args: 
        adj_pd (pd.DataFrame) : Pandas full-time adjacency matrix
    
    Returns:
        a (list) containing the reversed order of nodes
    """
    n_lags = max([int(node.split("_t-")[-1]) for node in adj_pd.columns if ("_t-" in node)])
    temp = []
    for t in reversed(range(1, n_lags + 1, 1)):
        temp.append(reversed(sorted([col for col in adj_pd.columns if f"_t-{str(t)}" in col])))
    temp.append(reversed(sorted([col for col in adj_pd.columns if (("_t" in col) and ("_t-" not in col))])))
    temp = [x for y in temp for x in y]

    return temp


def regular_order_pd(adj_pd: pd.DataFrame) -> list:
    """
    Returns the reversed order of the nodes of the Pandas full-time adjacency matrix, similar to the custom generator. 
    See the custom generator for details. Assumes the dataframe follows the node naming convention based on `_t-`.

    Args: 
        adj_pd (pd.DataFrame) : Pandas full-time adjacency matrix
    
    Returns:
        a (list) containing the reversed order of nodes
    """
    n_lags = max([int(node.split("_t-")[-1]) for node in adj_pd.columns if ("_t-" in node)])
    temp = [sorted([col for col in adj_pd.columns if (("_t" in col) and ("_t-" not in col))])]
    for t in range(1, n_lags + 1, 1):
        temp.append(sorted([col for col in adj_pd.columns if f"_t-{str(t)}" in col]))
    temp = [x for y in temp for x in y]

    return temp


def _from_full_to_lagged_adj(full_adj_pd: pd.DataFrame) -> torch.Tensor:
    """
    From full-time-graph to lagged adjacency matrix where `[j,i,l]>a` means variable `i` directly causes variable `j` with lag `\ell_max - l` with probability a. 
    Made as a separate method to avoid boilerplate code.

    Args: 
        full_adj_pd (pd.DataFrame) : the full-time-graph adjacency matrix as a `pd.DataFrame`

    Returns:
        lagged adjacency matrix, as a tensor of shape `(n_vars, n_vars, max_lag)`
    """
    # make sure that the nodes follow a regular lag ordering - i.e., grouped by lag and then in alphabetic order 
    full_adj_pd = full_adj_pd[regular_order_pd(full_adj_pd)]
    
    # get lagged nodes groups
    lag_dict = group_lagged_nodes(full_adj_pd.columns)
    n_vars = len(list(lag_dict.values())[0])
    n_lags = len(lag_dict) - 1

    # time-slices of full time graph
    slc_list = [lag_dict['0']]
    for lag in range(1, n_lags + 1, 1):
        slc_list.append(lag_dict[str(lag)])

    # create the lagged adjacency matrix
    adj = np.zeros(shape=(n_vars, n_vars, n_lags))
    for t, slc in enumerate(slc_list[::-1][:-1]):
        adj[:, :, t] = full_adj_pd.loc[slc, slc_list[0]].to_numpy().T

    return torch.tensor(adj)


def _from_lagged_adj_to_full(adj: np.ndarray, node_names=None) -> pd.DataFrame:

    """
    From lagged adjacency matrix to full-time-graph.
    Made as a separate method to avoid boilerplate code.

    Args: 
        adj (np.ndarray) : lagged adjacency matrix, as a NuPy array of shape `(n_vars, n_vars, max_lag)`.
        node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; if `None`, it follows an alphabetical order
    
    Returns: 
        temp_adj_pd: the full-time-graph adjacency matrix as a `pd.DataFrame`
    """
    # get intel
    if hasattr(adj, "detach"):
        adj = adj.detach().numpy()
    n_vars = adj.shape[1]
    n_lags = adj.shape[2]

    # Get default current node names if not provided
    if not node_names:
        node_names = list(string.ascii_uppercase)[:n_vars]
    t_node_names = [x + "_t" for x in node_names]

    # Get lagged node names and create the corresponding DataFrames
    lagged_adj_pd_list = []
    lagged_node_names_list = [t_node_names]
    for t in range(n_lags):
        # create the names
        lagged_node_names = [x + f"-{n_lags-t}" for x in t_node_names]
        lagged_node_names_list.append(lagged_node_names)
        # create the dataframe
        lagged_adj_pd = pd.DataFrame(data=adj[:, :, t].T, columns=t_node_names, index=lagged_node_names) 
        lagged_adj_pd_list.append(lagged_adj_pd)

    # Create the unrolled adjacency DataFrame
    sorted_names = list(sorted([y for x in lagged_node_names_list for y in x], key=lambda f: f.split("_t-")[-1] if "_t-" in f else "0"))
    temp_adj_pd = pd.DataFrame(
        data=np.zeros(shape=(len(sorted_names), len(sorted_names))), 
        columns=reversed(sorted_names), 
        index=reversed(sorted_names), 
        dtype=int
    )
    for lagged_adj_pd in lagged_adj_pd_list:
        for row in lagged_adj_pd.index:
            for col in lagged_adj_pd.columns:
                temp_adj_pd.loc[row, col] = lagged_adj_pd.loc[row, col]

    return temp_adj_pd


def _edges_for_causal_stationarity(temp_adj_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Takes as input a full-time graph adjacency matrix, checks which existing edges can be propagated through time,
    then propagates them. The aim is not to violate the causal consistency.

    *Note*: this is done separately during visualization, to mark the causal consistency edges w/ different colors on the fly.

    Args:
        - temp_adj_pd (pd.DataFrame) : a full-time graph adjacency matrix in a Pandas DataFrame format
    
    Return:
        - the initial full-time graph adjacency matrix w/ propagated edges in time in a Pandas DataFrame format
    """
    # from Pandas adjacency to NetworkX graph
    G = nx.from_pandas_adjacency(temp_adj_pd, create_using=nx.DiGraph)
    
    # lambda for getting the lag out of each node
    lbd_lag = lambda x: int(x.split('_t-')[-1]) if '_t-' in x else 0
    # lambda for getting the name out of each node
    lbd_name = lambda x: x.split('_t-')[0] if '_t-' in x else x.split('_t')[0]
    # add edges for causal stationarity
    added_edges = []
    for edge in G.edges:
        # calculate edge lag range   
        lag_range = lbd_lag(edge[0]) - lbd_lag(edge[1])
        if f"{edge[0].split('_t-')[0]}_t-{lbd_lag(edge[0]) + lag_range}" in G.nodes:
            G.add_edge(
                u_of_edge=f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range}", 
                v_of_edge=f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range}"
            )
            added_edges.append((
                f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range}", 
                f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range}"
            ))
    
    return nx.to_pandas_adjacency(G, dtype=int)


""" _____________________________________________ fMRI _____________________________________________ """

def from_fmri_to_lagged_adj(test_fmri: pd.DataFrame, label_fmri: pd.DataFrame) -> torch.Tensor:
    """
    Converts the fMRI pandas lagged edgelist to a lagged adjacency tensor.
    Assumes that the data ground truth edgelist has been read w/ column names: `['effect', 'cause', 'delay']`

    Args:
        test_fmri: the time-series data
        label_fmri: the ground truth Pandas edgelist

    Returns:
        the lagged adjacency matrix (of our used notation) as a torch.Tensor
    """ 
    # Check column names
    assert list(label_fmri.columns) == ['effect', 'cause', 'delay'], "Ground-truth edgelist read w/ wrong column names. \
Need to assign the following: ['effect', 'cause', 'delay']"

    # Find the number of lags
    n_lags = label_fmri['delay'].max()

    # Construct time-lagged adj matrix
    Y_fmri = np.zeros(shape=(test_fmri.shape[1], test_fmri.shape[1], n_lags))     # (dim, dim, time)
    for _ in label_fmri.index:
        Y_fmri[label_fmri['effect'], label_fmri['cause'], n_lags-label_fmri['delay']] = 1
    Y_fmri = torch.tensor(Y_fmri)

    return Y_fmri


""" _____________________________________________ CDML _____________________________________________ """

def from_cdml_to_lagged_adj(adj_pd: pd.DataFrame) -> torch.Tensor:
    """ 
    Converts an instance of CDML Pandas adjacency matrix to the lagged adjacency format. 

    Args: 
        adj_pd (pd.DataFrame): the CDML adjacency matrix
    
    Returns:
        adj (torch.Tensor): the lagged adjacency matrix
    """
    return _from_full_to_lagged_adj(adj_pd)


def y_from_cdml_to_lagged_adj(g_cdml: pd.DataFrame) -> torch.Tensor:
    """
    Another alias fro the `from_cdml_to_lagged_adj` function.
    Converts the ground truth adjacency matrix of CDML to the lagged adjacency format.
    
    Args:
        G: the pandas adjacency of the CDML ground truth graph

    Returns (torch.Tensor): 
        the corresponding lagged adjacency matrix as a PyTorch tensor object 
    """
    # Derive max lag and # vars from the adjacency matrix
    VAR = len([col for col in g_cdml if '_t-' not in col])
    LAG = max([int(col.split('_t-')[-1]) for col in g_cdml if '_t-' in col])
    # Intialize the adjacency matrix
    Y = np.zeros(shape=[VAR, VAR, LAG])
    # Fill it accordingly
    for idt in range(LAG):
        # Transpose it as notation expects first the effect then the cause
        slic = g_cdml.loc[[col for col in g_cdml.columns if f"_t-{LAG-idt}" in col],[col for col in g_cdml.columns if "-" not in col]].T 
        for idx, effect in enumerate(slic.index.to_list()):
            for idy, cause in enumerate(slic.columns.to_list()):
                if slic.loc[effect, cause]:
                    Y[(idx, idy, idt)] = 1

    return torch.tensor(Y)