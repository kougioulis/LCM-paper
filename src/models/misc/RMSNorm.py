import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, bias: bool=False, epsilon: float=1e-9):
        """
        Root Mean Square Layer Normalization (RMSNorm).
        Adapted from the original implementation by Zhang & Sennrich (2019) [*].
        RMSNorm simplifies LayerNorm by omiting mean centering and rescaling to unit variance.
        When mean of summed inputs is zero, it equals LayerNorm.
        
        Args:
        ---
            d (int): model size
            bias (bool): whether use bias term for RMSNorm. Defaults to 'False' as by default it does not enforce re-centering invariance
            epsilon (float): epsilon value. Defaults to 1e-9.
        
        Returns:
        ---
            (torch.tensor): Normalized tensor

        Notes:
        ---
        [*] Biao Zhang; Rico Sennrich (2019). Root Mean Square Layer Normalization. In Advances in Neural Information Processing Systems 32. Vancouver, Canada.
        https://github.com/bzhangGo/rmsnorm 
        """
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(d))

        self.bias = bias

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):

        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        x_normed = x / rms_x 

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed