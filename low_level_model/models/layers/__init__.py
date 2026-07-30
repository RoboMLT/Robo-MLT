"""Self-contained transformer layer utilities (attention, linear, RoPE).

These modules are pure-PyTorch with no external project dependencies and can be
imported anywhere without installing additional packages.
"""

from .attention import Attention
from .linear import MergedColumnLinear, QKVLinear
from .rope import RotaryEmbedding, apply_rotary_emb

__all__ = [
    "Attention",
    "QKVLinear",
    "MergedColumnLinear",
    "RotaryEmbedding",
    "apply_rotary_emb",
]
