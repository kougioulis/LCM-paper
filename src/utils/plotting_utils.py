import string
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.utils.transformation_utils import group_lagged_nodes, regular_order_pd


def plot_structure(
    temp_adj_pd: pd.DataFrame,
    node_color: Union[str, list, dict] = 'indianred',
    node_size: int = 1200,
    ax: plt.Axes = None,
    show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the causal structure of the model.

    Args
    ----
        temp_adj_pd (pd.DataFrame): 
            The base causal structure (without the causal stationarity edges; they are added on the fly here).
        node_color (str | list | dict): 
            Color of the nodes. If str, all nodes have that color. 
            If list, must match number of nodes. 
            If dict, maps base variable names to colors. 
            Default: 'indianred'.
        node_size (int): 
            Size of the nodes in the plot. Default: 1200.
        ax (matplotlib.axes._axes.Axes): 
            Axis object. Default: None.
        show (bool): 
            Whether to show the plot. Default: False.

    Returns
    ------
        f (matplotlib.figure.Figure): 
            The figure object, for potential further tempering.
        ax (matplotlib.axes._axes.Axes): 
            The axis object, for potential further tempering.
    """
    # from pandas to networkx
    G = nx.from_pandas_adjacency(temp_adj_pd, create_using=nx.DiGraph)

    # find the number of lags from the adjacency
    max_lag = max([int(node.split("_t-")[-1]) for node in G.nodes if "_t-" in node])

    # group nodes depending on their lags
    groups = {f"t-{lag}": [] for lag in reversed(range(max_lag + 1))}
    for node in G.nodes:
        for key in groups.keys():
            if key in node:
                groups[key].append(node)
        # fallback for ungrouped nodes
        if node not in [x for y in groups.values() for x in y]:
            groups[list(groups.keys())[-1]].append(node)

    # define figsize according to #nodes and #lags
    figsize = (
        max(3.2 * max_lag, 10),
        max(8, 1.2 * len(groups[list(groups.keys())[-1]]))
    )

    # define the nodes positions
    pos = {}
    x_current, y_current = 0, 0
    x_offset, y_offset = 3, 1
    for key in groups.keys():
        for node in groups[key]:
            pos[node] = (x_current, y_current)
            y_current -= y_offset
        x_current += x_offset
        y_current = 0

    # lambdas for parsing
    lbd_lag = lambda x: int(x.split('_t-')[-1]) if '_t-' in x else 0
    lbd_name = lambda x: x.split('_t-')[0] if '_t-' in x else x.split('_t')[0]

    # add edges for causal stationarity
    added_edges = []
    for edge in list(G.edges):  # avoid modifying while iterating
        lag_range = lbd_lag(edge[0]) - lbd_lag(edge[1])
        u_new = f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range}"
        v_new = f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range}"
        if u_new in G.nodes:
            G.add_edge(u_new, v_new)
            added_edges.append((u_new, v_new))

    # patch: assign positions to any new nodes
    for node in G.nodes:
        if node not in pos:
            pos[node] = (x_current, y_current)
            y_current -= y_offset

    # define edges for causal consistency
    edge_colors = {
        edge: "gray" if edge in added_edges else "black"
        for edge in G.edges
    }
    edge_color = list(edge_colors.values())

    # resolve node colors
    if isinstance(node_color, str):
        node_colors = [node_color] * len(G.nodes)
    elif isinstance(node_color, dict):
        node_colors = [
            node_color.get(lbd_name(node), 'indianred') for node in G.nodes
        ]
    elif isinstance(node_color, list):
        if len(node_color) != len(G.nodes):
            raise ValueError(
                f"node_color list length ({len(node_color)}) != number of nodes ({len(G.nodes)})"
            )
        node_colors = node_color
    else:
        raise TypeError(
            f"Invalid node_color type: {type(node_color)}. Must be str, list, or dict."
        )

    # draw it
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    else:
        f = ax.figure

    nx.draw(
        G, pos=pos, with_labels=True, ax=ax,
        node_size=node_size, node_color=node_colors, edge_color=edge_color,
        labels={
            node: "$" + lbd_name(node) + "_{t-" + str(lbd_lag(node)) + "}$" if "_t-" in node
            else f"${node}$" for node in G.nodes
        }
    )

    if show:
        plt.show()

    return f, ax


def plot_lagged_adjacency_structure(lagged_adj: torch.Tensor, node_names: list=None, node_color: str='indianred', node_size: int=1200, threshold: float=0.05,
                                    ax=None, show: bool=False, reduce: bool=True):
    """
    Creates a nx.DiGraph straight from the lagged adjacency structure and plots it.

    Args:
        lagged_adj (torch.Tensor): the lagged adjacency structure as a tensor of shape `(n_vars, n_vars, max_lag)`
        node_names (list): a list of strings with the names of the nodes, without any time index incorporated; 
                    if `None`, it follows an alphabetical order
        node_color (str): color of the nodes
        node_size (int): size of the nodes
        threshold (float): value above which an edge is added to the graph
        ax (matplotlib.Axes): the axes to plot on
        show (bool): whether to show the plot
        reduce (bool): whether to reduce the graph to the nodes with non-zero lagged relationships

    Returns:
        temp_adj_pd (pd.DataFrame): the adjacency matrix of the graph as a `pd.DataFrame`
    """

    temp_adj_pd, _ = process_lagged_adj(
        lagged_adj=lagged_adj, 
        threshold=threshold, 
        node_names=node_names, 
        reduce=reduce
    )

    # plot and return
    _ = plot_structure(temp_adj_pd=temp_adj_pd, ax=ax, node_color=node_color, node_size=node_size, show=show)
  
    return temp_adj_pd


def process_lagged_adj(
        lagged_adj: torch.Tensor, 
        threshold: float = 0.05, 
        node_names: list = None, 
        reduce: bool = True
) -> list:
    """
    Processes the lagged adjacency structure and returns the corresponding `nx.DiGraph` object.
    
    Args
    ----
        lagged_adj (torch.Tensor):
            The lagged adjacency structure.
        threshold (float):
            The threshold above which the links are considered to be causal. Default is `0.05`.
        node_names (list):
            The names of the variables. If `None`, the default ones are used.
        reduce (bool):
            Whether to reduce the adjacency matrix. Default is `True`.

    Returns
    -------
        temp_adj_pd (pd.DataFrame):
            The corresponding adjacency matrix.
        lagged_nodes (list):gg
            The names of the nodes.
    """
  
    if hasattr(lagged_adj, "numpy"):
        lagged_adj = lagged_adj.numpy().copy()

    # threshold the adjacency matrix
    lagged_adj[lagged_adj < threshold] = 0
    lagged_adj[lagged_adj >= threshold] = 1

    # get the number of lags and variables
    n_vars = lagged_adj.shape[1]
    n_lags = lagged_adj.shape[2]

    # if names are not provided get the default ones
    if node_names is None:
        node_names = list(string.ascii_uppercase)[:n_vars]
  
    # create the lagged variables and name the nodes accordingly
    lagged_nodes = [node + f"_t-{t}" for t in range(n_lags+1) for node in node_names]
    lagged_nodes = [node.replace("_t-0", "_t") for node in lagged_nodes]
    lag_dict = group_lagged_nodes(lagged_nodes=lagged_nodes)
  
    # instantiate an empty nx.DiGraph object
    G = nx.from_pandas_adjacency(
        df=pd.DataFrame(
            columns=lagged_nodes, 
            index=lagged_nodes, 
            data=np.zeros(shape=(len(lagged_nodes), len(lagged_nodes)))
        ), 
        create_using=nx.DiGraph
    )

    # add edges
    for i in range(lagged_adj.shape[0]):
        for j in range(lagged_adj.shape[1]):
            for t in range(lagged_adj.shape[2]):
                if lagged_adj[i, j, t]==1:
                    lag = str(n_lags-t)
                    effect = lag_dict['0'][i]
                    cause = lag_dict[lag][j]
                    # print(f"{cause} -> {effect} ({lag})")
                    G.add_edge(u_of_edge=cause, v_of_edge=effect)

    # create a pandas adjacency
    adj_pd=nx.to_pandas_adjacency(G, dtype=int)

    # drop redundant elements
    # if reduce:
    node2drop = [col for col in adj_pd.columns if adj_pd.loc[:, col].sum()>0] + \
    [col for col in adj_pd.columns if adj_pd.loc[col, :].sum()>0]
    adj_pd = adj_pd.drop(index=[col for col in adj_pd.columns if col not in node2drop]).\
                drop(columns=[col for col in adj_pd.columns if col not in node2drop])
  
    # find the (new) number of lags from the adjacency
    max_lag = max([int(node.split("_t-")[-1]) for node in adj_pd.columns if "_t-" in node])

    # group (new) nodes depending on their lags
    groups = {f"t-{lag}":[] for lag in reversed(range(max_lag + 1))}
    for node in adj_pd.columns:
        for key in groups.keys():
            if key in node:
                groups[key].append(node)
        if node not in [x for y in groups.values() for x in y]:
            groups[list(groups.keys())[-1]].append(node)

    # define figsize according to (new) #nodes and #lags
    figsize = (max([3.2 * max_lag, 10]), max([8, 1.2 * len(groups[list(groups.keys())[-1]])]))

    # returned info: 1. reduced adj_pd, 2. estimated figsize
    return adj_pd, figsize


def plot_comparison(
        label_lagged: torch.Tensor, 
        pred_lagged: torch.Tensor,  
        threshold: float = 0.05,
        node_names: list = None, 
        color_palette: str ='blend:#7AB,#EDA', 
        node_size: int = 1200,
        minimal: bool = True
) -> None:
    """ 
    Plots a comparative plot of i) the predicted lagged causal graph and ii) the ground truth causal graph, side-to-side.

    Args:
        label_lagged (torch.Tensor): the ground truth lagged adjacency tensor
        pred_lagged (torch.Tensor): the predicted lagged adjacency tensor
        threshold (float): the threshold used to binarize the adjacency matrix. Defalt is `0.05`.
        node_names (list): Optional list of node names. Default is `None`.
        color_palette (str): Color palette for the plot. Default is ` `.
        node_size (int): Size of the plotted nodes. Default is `1200`.
        minimal (bool): Whether to create a minimal plot. Default is `True`. 

    Returns:
        f (matplotlib.figure.Figure) :the figure object, for potential further tempering
        ax (matplotlib.axes._axes.Axes) :the axis object, for potential further tempering
    """

    # Process lagged adjacencies into pandas adjacency matrices
    label_pd, label_figsize = process_lagged_adj(
        lagged_adj=label_lagged, 
        threshold=threshold, 
        node_names=node_names, 
        reduce=True
    )
    pred_pd, pred_figsize = process_lagged_adj(
        lagged_adj=pred_lagged, 
        threshold=threshold, 
        node_names=node_names, 
        reduce=True
    )

    # catch the case where the prediction does not include nodes from the label after processing
    for col in label_pd.columns:
        if col not in pred_pd.columns:
            pred_pd.loc[:, col] = 0.0
            pred_pd.loc[col, :] = 0.0

    pred_pd = pred_pd.loc[regular_order_pd(pred_pd), regular_order_pd(pred_pd)] 

    # NOTE: By definition, the label graph's nodes is a subset of the predicted graph's nodes;
    # therefore, prediction's node set is reduced for better visualization
    if minimal:
        pred_pd = pred_pd.loc[label_pd.columns, label_pd.columns].copy()
        figsize = label_figsize
    else:
        figsize = (max(list(zip(label_figsize, pred_figsize))[0]), max(list(zip(label_figsize, pred_figsize))[1]))

    # colormap: include all unique unlagged variables from both graphs
    all_cols = list(set(label_pd.columns) | set(pred_pd.columns))
    unlagged_nodes = sorted(set(col.split("_t")[0] for col in all_cols))

    # ensure enough colors
    palette = sns.color_palette(color_palette, n_colors=len(unlagged_nodes))
    colormap = dict(zip(unlagged_nodes, palette))

    # build node color lists
    label_node_color = [colormap[col.split("_t")[0]] for col in label_pd.columns]
    pred_node_color = [colormap[col.split("_t")[0]] for col in pred_pd.columns]

    # sanity check
    assert len(label_pd.columns) == len(label_node_color), "Mismatch in label graph nodes and colors!"
    assert len(pred_pd.columns) == len(pred_node_color), "Mismatch in pred graph nodes and colors!"

    # Plot both graphs
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})
    _ = plot_structure(temp_adj_pd=label_pd, ax=axs[0], node_color=label_node_color, node_size=node_size)
    _ = plot_structure(temp_adj_pd=pred_pd, ax=axs[1], node_color=pred_node_color, node_size=node_size)

    axs[0].set_title("Ground Truth Causal Graph", fontsize=18)
    axs[1].set_title("Discovered Causal Graph", fontsize=18)
    plt.tight_layout()
    plt.show()

    return f, axs


def plot_adjacency_heatmaps(pred_adj: torch.Tensor, true_adj: torch.Tensor, absolute_errors: bool=False, export: bool=False) -> None:
    """
    Plot per-lag heatmaps comparing predicted and ground truth lagged adjacency matrices.

    Args:
        pred_adj: torch.Tensor or np.ndarray, shape `(num_vars, num_vars, max_lag)`
        true_adj: torch.Tensor or np.ndarray, same shape
        model_name: str, for plot title

    Returns:
        None
    """
    if hasattr(pred_adj, 'detach'):
        pred_adj = pred_adj.detach().cpu().numpy()
    if hasattr(true_adj, 'detach'):
        true_adj = true_adj.detach().cpu().numpy()

    num_vars, _, max_lag = pred_adj.shape

    plt.figure(figsize=(6 * max_lag, 12))

    for lag in range(max_lag):
        pred_mat = pred_adj[:, :, lag]
        true_mat = true_adj[:, :, lag]
        error_mat = np.abs(pred_mat - true_mat)

        row_labels = [f"V{i+1}" for i in range(num_vars)]
        col_labels = [f"V{j+1}" for j in range(num_vars)]

        # Ground Truth
        plt.subplot(3, max_lag, lag + 1)
        sns.heatmap(
            true_mat, vmin=0, vmax=1, cmap="viridis",
            xticklabels=col_labels, yticklabels=row_labels,
            linewidths=0.5, linecolor='white', square=True,
            cbar=True, annot=True, fmt=".2f"
        )
        #plt.title(f"Ground Truth - Lag {lag+1}", fontsize=13)
        plt.title(f"Ground Truth - Lag {max_lag - lag}", fontsize=13)

        plt.xlabel("Parent Nodes", fontsize=11)
        plt.ylabel("Target Nodes", fontsize=11)

        # Prediction
        plt.subplot(3, max_lag, max_lag + lag + 1)
        sns.heatmap(
            pred_mat, vmin=0, vmax=1, cmap="viridis",
            xticklabels=col_labels, yticklabels=row_labels,
            linewidths=0.5, linecolor='white', square=True,
            cbar=True, annot=True, fmt=".2f"
        )
        #plt.title(f"Prediction - Lag {lag+1}", fontsize=13)
        plt.title(f"Prediction - Lag {max_lag - lag}", fontsize=13)

        plt.xlabel("Parent Nodes", fontsize=11)
        plt.ylabel("Target Nodes", fontsize=11)

        # Absolute Error
        if absolute_errors:
            plt.subplot(3, max_lag, 2 * max_lag + lag + 1)
            sns.heatmap(
                error_mat, vmin=0, vmax=1, cmap="viridis",
                xticklabels=col_labels, yticklabels=row_labels,
                linewidths=0.5, linecolor='white', square=True,
                cbar=True, annot=True, fmt=".2f"
            )
            #plt.title(f"Abs Error - Lag {lag+1}", fontsize=13)
            plt.title(f"Abs Error - Lag {max_lag - lag}", fontsize=13)
            plt.xlabel("Parent Nodes", fontsize=11)
            plt.ylabel("Target Nodes", fontsize=11)

    plt.tight_layout()
    plt.show()

    if export:
        # save as pdf
        plt.savefig("adjacency_heatmaps.pdf", format="pdf")


def plot_adjacency_matrices(pred_adj: torch.Tensor, true_adj: torch.Tensor) -> None:
    """
    A more plain function compared to plot_adjacency_heatmaps, for comparing the predicted lagged adjacency matrices against the ground truth.

    Args:
        pred_adj: torch.Tensor or np.ndarray, shape `(num_vars, num_vars, max_lag)`
        true_adj: torch.Tensor or np.ndarray, same shape

    Returns:
        None
    """
    n_lags = true_adj.shape[2]
    for lag in range(n_lags):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(true_adj[:, :, lag], cmap='Reds', vmin=0, vmax=1)
        plt.title(f"True Adjacency (lag {lag})")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(pred_adj[:, :, lag], cmap='Blues', vmin=0, vmax=1)
        plt.title(f"Output (lag {lag})")
        plt.colorbar()
        plt.show()


def _draw_styled_dag(adj_pd, ax, node_colors, node_size=1400, title=None):
    """
    Helper function for fancy drawing of a lagged causal graph"""
    G = nx.from_pandas_adjacency(adj_pd, create_using=nx.DiGraph)

    # layout: group by time-lag for block structure
    pos = {}
    columns = list(adj_pd.columns)
    groups = {}
    for col in columns:
        base, _, lag = col.partition("_t-")
        lag = int(lag) if lag.isdigit() else 0
        groups.setdefault(lag, []).append(col)

    x_off = 2.0
    y_off = 1.3
    #for i, lag in enumerate(sorted(groups.keys())):
    for i, lag in enumerate(sorted(groups.keys(), reverse=True)):
        ys = np.linspace(0, -y_off*(len(groups[lag])-1), len(groups[lag]))
        for y, node in zip(ys, groups[lag]):
            pos[node] = (i * x_off, y)

    # classify edges
    self_edges = []
    cross_edges = []
    for u, v in G.edges():
        if u.split("_t")[0] == v.split("_t")[0]:
            self_edges.append((u, v))
        else:
            cross_edges.append((u, v))

    # draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=node_size,
        node_color=node_colors, edgecolors="black", linewidths=0.8,
        ax=ax, alpha=0.9
    )

    # draw edges
    nx.draw_networkx_edges(
        G, pos, edgelist=cross_edges,
        arrows=True, arrowstyle='-|>',
        arrowsize=40,
        width=2.0,
        edge_color='black',
        connectionstyle='arc3,rad=0.00',
        min_source_margin=5.50,  # distance from source node
        min_target_margin=5.50,  # distance from target node
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=self_edges,
        arrows=True, arrowstyle="-|>", arrowsize=40,
        width=1.8, edge_color="gray", alpha=0.4,
        connectionstyle="arc3,rad=0.0",
        min_source_margin=5.5,
        min_target_margin=5.5,
        ax=ax
    )

    # draw labels as latex: X^{i}_{t-k}
    labels = {}
    for node in G.nodes:
        var_name, _, lag = node.partition("_t-")
        lag = int(lag) if lag.isdigit() else 0
        if lag == 0:
            labels[node] = rf"$\mathrm{{{var_name}}}$"

        else:
            labels[node] = rf"${var_name}_{{t-{lag}}}$"
            labels[node] = rf"$\mathrm{{{var_name}}}_{{t-{lag}}}$"

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)

    # formatting
    ax.set_title(title, fontsize=15, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.3)


def plot_comparison_fancy(
        label_lagged: torch.Tensor, 
        pred_lagged: torch.Tensor,  
        threshold: float = 0.05,
        X: torch.Tensor = None,
        node_names: list = None, 
        color_palette: str ='tab10', 
        node_size: int = 1400,
        minimal: bool = True,
        max_ts: int = 150
):
    """
    Fancy, paper-grade comparison plot of i) the predicted lagged causal graph and ii) the ground truth causal graph.
    Also plots the input time series X, if provided. Based on `plot_comparison`.
    """

    # --- convert tensors to adjacency pandas ---
    label_pd, label_figsize = process_lagged_adj(
        lagged_adj=label_lagged, 
        threshold=threshold, 
        node_names=node_names, 
        reduce=True
    )
    pred_pd, pred_figsize = process_lagged_adj(
        lagged_adj=pred_lagged, 
        threshold=threshold, 
        node_names=node_names, 
        reduce=True
    )

    # align prediction to truth if needed
    for col in label_pd.columns:
        if col not in pred_pd.columns:
            pred_pd.loc[:, col] = 0
            pred_pd.loc[col, :] = 0

    pred_pd = pred_pd.loc[regular_order_pd(pred_pd), regular_order_pd(pred_pd)]

    if minimal:
        pred_pd = pred_pd.loc[label_pd.columns, label_pd.columns].copy()

    #print(f'label_pd column names: {label_pd.columns}')
    #print(f'pred_pd column names: {pred_pd.columns}')

    # color palette
    import seaborn as sns
    unlagged = sorted(set(col.split("_t")[0] for col in label_pd.columns))
    palette = sns.color_palette(color_palette, n_colors=len(unlagged))
    color_map = dict(zip(unlagged, palette))

    label_colors = [color_map[col.split("_t")[0]] for col in label_pd.columns]
    pred_colors = [color_map[col.split("_t")[0]] for col in pred_pd.columns]

    # figure mosaic: left GT, middle Pred, right TS
    fig = plt.figure(figsize=(16, 8))
    mosaic = """
    AB
    AC
    """
    axd = fig.subplot_mosaic(mosaic, width_ratios=[1.2, 1.4])
    
    ax_gt = axd["A"]
    ax_pred = axd["B"]
    ax_ts = axd["C"]

    # draw styled DAGs
    _draw_styled_dag(label_pd, ax_gt, label_colors, node_size=node_size, title="Ground Truth Causal Graph")
    _draw_styled_dag(pred_pd, ax_pred, pred_colors, node_size=node_size, title="Discovered Causal Graph")

    # time series panel
    if X is not None:
        X = X.detach().cpu().numpy() if hasattr(X, "detach") else X
        T = min(len(X), max_ts)
        t = np.arange(T)
        for i, var in enumerate(unlagged):
            ax_ts.plot(t, X[:T, i], lw=1.4, color=color_map[var], label=var)
        ax_ts.set_title("Sample Time Series", fontsize=14)
        #ax_ts.set_xlabel("Time", fontsize=10)
        ax_ts.grid(alpha=0.25, lw=0.4)
        ax_ts.set_yticks([])
        ax_ts.set_xticks([])
        ax_ts.spines['top'].set_alpha(0.5)
        ax_ts.spines['right'].set_alpha(0.5)
        ax_ts.spines['left'].set_alpha(0.5)
        ax_ts.spines['bottom'].set_alpha(0.5)
        ax_ts.tick_params(labelsize=8)

    fig.tight_layout(pad=0.8)
    plt.show()

    return fig