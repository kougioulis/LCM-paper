import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum


class RotaryEmbedding(nn.Module):
    def __init__(self, d_emb: int):
        """
        Rotary Position Embedding (RoPE) for transformer models.
        Implements relative positional encoding through rotation matrices.
        
        Args:
        ---
            d_emb (int): embedding dimension (must be even)
        
        Notes:
        ---
        [*] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: 
            Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864
        """
        super().__init__()
        if d_emb % 2 == 1:
            raise ValueError("Embedding dimension must be even.")

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_emb, 2).float() / d_emb))
        self.register_buffer("inv_freq", inv_freq)
    

    def forward(self, seq_len: int, device: torch.device):
        """
        Args:
        ---
            seq_len (int): sequence length
            device (torch.device): compute device
            
        Returns:
        ---
            (torch.tensor): rotary positional embeddings of shape [max_seq_len, dim]
        """
        seq = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)

        return torch.cat((freqs, freqs), dim=-1)

# helpers 
def rotate_half(x: torch.tensor):
    """
    Rotates half the channels by swapping and negating
    Used in rotary position embedding calculation
    
    Args:
    ---
        x (torch.tensor): input tensor of shape (1,d) (d must be even)
        
    Returns:
    ---
        (torch.tensor): rotated version of input
    """
    if x.shape[1] % 2 == 1:
        raise ValueError("Input dimension must be even.")

    x1, x2 = x.chunk(2, dim=-1)

    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos: torch.tensor, t: torch.tensor):
    """
    Applies rotary positional embeddings to input tensor
    
    Args:
    ---
        pos (torch.tensor): positional embeddings of shape [seq_len, dim]
        t (torch.tensor): input tensor of shape [batch, heads, seq_len, dim]
        
    Returns:
    ---
        torch.tensor: position-aware transformed tensor
    """
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())
