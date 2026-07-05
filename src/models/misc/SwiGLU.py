import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        SwiGLU (Swish Gated Linear Unit) activation layer
        
        Args:
        ---
            dim (int): input dimension
            hidden_dim (int): hidden dimension (typically 4x input dim)
            
        Notes:
        ---
        [*] Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        """
        Forward pass for SwiGLU activation
        
        Args:
        ---
            x (torch.tensor): input tensor of shape [..., dim] 
            
        Returns:
        ---
            (torch.tensor): output tensor of shape [..., dim]
        """
        x, gate = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x)