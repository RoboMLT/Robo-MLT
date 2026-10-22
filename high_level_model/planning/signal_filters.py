"""Optional causal filters for the System-2 head probabilities.

The method thresholds the raw probabilities with a k-of-k debounce (paper Sec. III-B), so the
deployed configs leave these filters off (``median_window=1, ema_alpha=1.0``). They are kept for
offline analysis and plotting, where a lightly smoothed curve is easier to read. The module is
torch-free so the planning layer stays importable without a GPU.

Filter chain:  causal median (removes single-sample spikes) -> EMA -> optional monotone clamp.
"""

from collections import deque
from statistics import median
from typing import Optional

__all__ = ["SignalFilter"]


class SignalFilter:
    """Causal filter: median spike removal -> EMA -> optional monotone clamp.

    Args:
        median_window: window for the causal median (``1`` disables it).
        ema_alpha: EMA weight on the new (median) sample; ``1.0`` disables smoothing. The EMA is
            initialised to the first sample, so a constant input passes through with zero lag.
        monotone: if True, the output never decreases. The clamp is applied after smoothing so a
            single noisy spike cannot ratchet the value up permanently.
    """

    def __init__(self, median_window: int = 1, ema_alpha: float = 1.0,
                 monotone: bool = False) -> None:
        self.median_window = max(1, int(median_window))
        self.ema_alpha = float(ema_alpha)
        self.monotone = bool(monotone)
        self._buf: deque = deque(maxlen=self.median_window)
        self._ema: Optional[float] = None
        self._out: Optional[float] = None

    def reset(self) -> None:
        """Clear all temporal state (call on skill switch / new episode)."""
        self._buf.clear()
        self._ema = None
        self._out = None

    def update(self, x: float) -> float:
        """Feed one raw value, return the filtered value."""
        x = float(x)
        self._buf.append(x)
        med = median(self._buf)  # window-of-1 => identity
        if self._ema is None:
            self._ema = med
        else:
            self._ema = self.ema_alpha * med + (1.0 - self.ema_alpha) * self._ema
        out = self._ema
        if self.monotone and self._out is not None:
            out = max(self._out, out)
        self._out = out
        return out
