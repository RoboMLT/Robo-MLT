"""Fused Linear Layers for Transformer Optimization.

This module provides fused linear projections that combine multiple
operations into a single matrix multiplication for efficiency:

- QKVLinear: Fused Q/K/V projection for attention
- MergedColumnLinear: Fused gate/up projection for MLP
"""

import torch
import torch.nn.functional as F
from torch import nn


class QKVLinear(nn.Module):
    """Fused Query-Key-Value projection for multi-head attention.

    Combines Q, K, V projections into a single matrix multiply:
        [Q, K, V] = x @ W^T  where W = [W_q; W_k; W_v]
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads

        output_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V tensors.

        Args:
            x: Input tensor [B, L, hidden_size].

        Returns:
            q: [B, num_heads, L, head_size].
            k: [B, num_kv_heads, L, head_size].
            v: [B, num_kv_heads, L, head_size].
        """
        if x.dim() != 3:
            raise ValueError(f"QKVLinear expects 3D input [B, L, D], got {x.shape}")

        bsz, seqlen, _ = x.shape
        out = F.linear(x, self.weight, self.bias)

        total_heads = self.num_heads + 2 * self.num_kv_heads
        out = out.view(bsz, seqlen, total_heads, self.head_size)
        out = out.permute(0, 2, 1, 3).contiguous()

        q = out[:, : self.num_heads]
        k = out[:, self.num_heads : self.num_heads + self.num_kv_heads]
        v = out[:, self.num_heads + self.num_kv_heads :]
        return q, k, v


class MergedColumnLinear(nn.Module):
    """Fused column-parallel linear layer for MLP.

    Combines multiple projections (e.g., gate/up in SwiGLU) into one matmul:
        [gate, up] = x @ W^T  where W = [W_gate; W_up]
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = list(output_sizes)

        output_size = sum(self.output_sizes)
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = F.linear(x, self.weight, self.bias)
        return torch.split(out, self.output_sizes, dim=-1)


__all__ = ["QKVLinear", "MergedColumnLinear"]
