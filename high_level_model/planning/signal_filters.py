"""Causal output-side filters for the System-2 completion gate signals.

The gate emits a raw per-call ``progress``/``back_prob`` sigmoid with no temporal
state, so the curve the pipeline thresholds against is jittery. These filters
de-jitter the *value* (variance reduction) before the pipeline's k-of-k decision
debouncing; they are intentionally **torch-free** (pure Python) so the planning
layer stays importable/testable without a GPU.

Pipeline:  median(remove single-frame spikes) -> EMA(smooth) -> optional monotone
clamp(progress only never regresses within a subtask segment).
"""

from collections import deque
from statistics import median
from typing import Optional

__all__ = ["ProgressSignalFilter"]


class ProgressSignalFilter:
    """Causal filter: median spike-removal -> EMA -> optional monotone clamp.

    Args:
        median_window: window for the causal median (``1`` disables it).
        ema_alpha: EMA weight on the new (median) sample; ``1.0`` disables smoothing.
            The EMA is initialised to the first sample, so a *constant* input passes
            through with zero lag (this keeps the unit tests deterministic).
        monotone: if True, the output never decreases — semantically correct for
            within-segment progress and the cheapest cure for downward jitter. The
            clamp is applied *after* smoothing so a single noisy spike can't ratchet
            the value up permanently.
    """

    def __init__(self, median_window: int = 3, ema_alpha: float = 0.4,
                 monotone: bool = True) -> None:
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
            self._ema = med  # init to first sample => constant input has zero lag
        else:
            self._ema = self.ema_alpha * med + (1.0 - self.ema_alpha) * self._ema
        out = self._ema
        if self.monotone and self._out is not None:
            out = max(self._out, out)
        self._out = out
        return out
