import numpy as np
import pandas as pd
import torch

def y_from_cdml_to_lagged_adj(g_cdml: pd.DataFrame):
    """
    Converts the ground truth adjacency matrix of CDML datasets to a lagged adjacency format.

    Arguments:
        g_cdml (pandas.DataFrame): the pandas adjacency of the CDML ground truth graph

    Returns (torch.tensor): 
        The corresponding lagged adjacency matrix of CP, as a PyTorch tensor object 
    """
    # Derive max lag and number of vars from the adjacency matrix
    VAR = len([col for col in g_cdml if '_t-' not in col])
    LAG = max([int(col.split('_t-')[-1]) for col in g_cdml if '_t-' in col])
    Y = np.zeros(shape=[VAR, VAR, LAG])

    for idt in range(LAG):
        # transpose it as the format expects first the effect and then the cause
        slic = g_cdml.loc[[col for col in g_cdml.columns if f"_t-{LAG-idt}" in col],[col for col in g_cdml.columns if "-" not in col]].T 
        for idx, effect in enumerate(slic.index.to_list()):
            for idy, cause in enumerate(slic.columns.to_list()):
                if slic.loc[effect, cause]:
                    Y[(idx, idy, idt)] = 1

    return torch.tensor(Y)