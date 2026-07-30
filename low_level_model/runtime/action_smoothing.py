"""Output-side action-chunk smoothing for the System-1 async executor.

PI0 is a multi-modal flow policy: consecutive action chunks are sampled
*independently*, so at a chunk boundary the executor's hard switch from one chunk
to the next produces an out-of-distribution discontinuity — the visible "jerk"
between chunks. A naive low-pass / EMA filter on the action stream would suppress
it only by adding phase lag (≈ the filter time constant) and smearing legitimate
fast/contact motions, because the jitter is a *boundary discontinuity*, not
high-frequency noise.

:class:`TemporalChunkBlender` instead acts only at the join, mirroring the
"temporal chunk-wise smoothing" of χ₀ (Yu et al. 2026, Algorithm 1; reference
implementation ``StreamActionBuffer.integrate_new_chunk`` in
https://github.com/OpenDriveLab/kai0). It (1) drops the stale leading steps of the
incoming chunk that overlap, in wall-clock time, with steps already executed
during inference (latency compensation), then (2) linearly cross-fades the
overlap window between the residual tail of the old chunk and the new chunk.
Steps away from the boundary are untouched, so there is no steady-state lag, and
a genuine motion change still propagates fully within the cross-fade window.

The module is intentionally **numpy-only** (no torch / lerobot) so it stays
importable and unit-testable without a GPU, like
``high_level_model/planning/signal_filters.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["TemporalChunkBlender", "JointRateLimiter", "interpolate_action"]


class TemporalChunkBlender:
    """Cross-fade an incoming action chunk into the residual tail of the current one.

    Args:
        latency_k: Maximum number of leading steps of the new chunk to discard as
            stale (the inference-delay compensation ``max_k`` in χ₀). Set this to
            the measured inference delay ``d ≈ ceil(latency * fps)`` (in control
            steps). ``0`` disables the drop entirely — note the reference repo's
            default is ``0``, i.e. a no-op, so this must be set deliberately or the
            blend will replay already-executed steps and re-introduce a jump.
        min_smooth_steps: Minimum cross-fade window length (``min_m`` in χ₀). When
            the residual old tail is shorter than this it is padded by repeating its
            last command (or, at a clean boundary, the last executed action), so the
            new chunk is always ramped in over at least this many steps.
        weight_profile: Cross-fade ramp. ``"linear"`` (default, matches χ₀) goes
            100% old → 100% new linearly; ``"smoothstep"`` uses a C¹ ramp
            (Hy-Embodied-0.5 Bézier-style) for a gentler velocity profile.
    """

    def __init__(self, latency_k: int = 0, min_smooth_steps: int = 8,
                 weight_profile: str = "linear") -> None:
        self.latency_k = max(0, int(latency_k))
        self.min_smooth_steps = max(1, int(min_smooth_steps))
        if weight_profile not in ("linear", "smoothstep"):
            raise ValueError(f"weight_profile must be 'linear' or 'smoothstep', got {weight_profile!r}")
        self.weight_profile = weight_profile

    def _old_weights(self, length: int) -> np.ndarray:
        """Per-step weight on the *old* sequence over the overlap (1.0 → 0.0)."""
        if length <= 1:
            return np.ones(max(length, 1), dtype=float)
        x = np.linspace(0.0, 1.0, length, dtype=float)
        if self.weight_profile == "smoothstep":
            x = x * x * (3.0 - 2.0 * x)
        return 1.0 - x

    def merge(
        self,
        old_remaining: Optional[np.ndarray],
        new_chunk: np.ndarray,
        consumed: int,
        last_action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Blend ``new_chunk`` into the residual ``old_remaining``.

        Args:
            old_remaining: Not-yet-executed tail of the current chunk
                ``[m, action_dim]``, or ``None``/empty at a clean boundary.
            new_chunk: Freshly inferred chunk ``[n, action_dim]``.
            consumed: Control steps elapsed since the new chunk's inference was
                launched (i.e. how many of its leading steps are now stale). The
                actual drop is ``min(consumed, latency_k)``.
            last_action: Last command actually published. Used to seed the cross-fade
                when ``old_remaining`` is empty so the new chunk still ramps in from
                where the robot is, rather than jumping.

        Returns:
            The merged buffer ``[M, action_dim]`` to execute next (variable length:
            ``M = len(old overlap) + len(new) - overlap - drop``).
        """
        # Preserve the policy's action dtype (typically float32). Upcasting to
        # float64 here would flow back into the model via ``future_state`` (the
        # executor feeds the buffer's last row in as ``observation.state``), and a
        # float64 state silently fails the dtype guard of a torch.compiled
        # (dynamic=False) policy -> a recompile on every step that looks like a hang.
        new = np.asarray(new_chunk)
        dtype = new.dtype if np.issubdtype(new.dtype, np.floating) else np.float32
        new = new.astype(dtype, copy=False)
        if new.ndim == 1:
            new = new.reshape(1, -1)
        if len(new) == 0:
            raise ValueError("new_chunk must be non-empty")

        # 1) Drop stale leading steps (latency compensation). Always keep >=1 row so
        #    we have something executable (χ₀ skips the update; we cannot, since the
        #    old chunk may already be exhausted).
        drop_n = min(max(0, int(consumed)), self.latency_k, len(new) - 1)
        new = new[drop_n:]

        # 2) Build the old sequence to cross-fade against.
        old: Optional[np.ndarray] = None
        if old_remaining is not None and len(old_remaining) > 0:
            old = np.asarray(old_remaining, dtype=dtype)
            if old.ndim == 1:
                old = old.reshape(1, -1)
            if len(old) < self.min_smooth_steps:  # pad short tail by repeating its last command
                pad = np.repeat(old[-1:], self.min_smooth_steps - len(old), axis=0)
                old = np.concatenate([old, pad], axis=0)
        elif last_action is not None:  # clean boundary: seed from where the robot is
            la = np.asarray(last_action, dtype=dtype).reshape(1, -1)
            old = np.repeat(la, self.min_smooth_steps, axis=0)

        if old is None:  # bootstrap / nothing to blend against
            return new

        # 3) Linear (or smoothstep) cross-fade over the overlap window (keep dtype).
        overlap = min(len(old), len(new))
        w_old = self._old_weights(overlap).reshape(-1, 1).astype(dtype)
        smoothed = w_old * old[:overlap] + (1.0 - w_old) * new[:overlap]
        return np.concatenate([smoothed, new[overlap:]], axis=0).astype(dtype, copy=False)


def interpolate_action(prev_action: np.ndarray, cur_action: np.ndarray,
                       max_step: np.ndarray | float) -> np.ndarray:
    """Sub-divide a jump from ``prev_action`` to ``cur_action`` into bounded steps.

    Optional publish-time velocity clamp (ported from χ₀'s ``interpolate_action``):
    any per-dimension delta larger than ``max_step`` is split into a linear ramp so
    no single control step commands a jump bigger than ``max_step``. Independent of
    the boundary blend; use as a cheap safety net on the final command stream.

    Args:
        prev_action: Last published action ``[action_dim]``.
        cur_action: Target action ``[action_dim]``.
        max_step: Max allowed per-dimension change per control step (scalar or
            ``[action_dim]``).

    Returns:
        Array ``[k, action_dim]`` of intermediate actions ending at ``cur_action``
        (``k == 1`` — just ``cur_action`` — when no dimension exceeds ``max_step``).
    """
    prev = np.asarray(prev_action, dtype=float)
    cur = np.asarray(cur_action, dtype=float)
    steps = np.asarray(max_step, dtype=float)
    n = int(np.max(np.ceil(np.abs(cur - prev) / np.maximum(steps, 1e-8))))
    if n <= 1:
        return cur[np.newaxis, :]
    return np.linspace(prev, cur, n + 1)[1:]

class JointRateLimiter:
    """Publish-time per-joint velocity (rate) limiter for the command stream.

    Clamps the per-step change of each commanded joint to ``max_step`` (same units as
    the action — radians for the Piper arm joints). This bounds the commanded joint
    velocity to ``max_step * fps`` and turns any chunk-boundary discontinuity into a
    bounded ramp the hardware can physically track. It is the software equivalent of
    the accel-limited trajectory smoothing that a high-stiffness controller (Piper
    **MIT mode**) does *not* perform internally: in MIT mode every commanded target is
    chased directly at high gain, so a raw inter-chunk jump becomes a physical jerk;
    the limiter caps how far a single command may move.

    Unlike an EMA / low-pass, this is a *governor*, not a filter: when the target
    stream already moves within the limit it is passed through **unchanged**, so there
    is no steady-state lag and legitimate fast motions are not smeared — only steps
    that exceed the limit are rate-capped (the single-step form of χ₀'s
    ``interpolate_action`` clamp). Being a pure output stage, it is deliberately
    decoupled from the executor's blender seed / future-state, so the smoothing stages
    compose without feeding back into the policy.

    Args:
        max_step: Max allowed per-step |Δ| for each joint. A scalar broadcasts to
            every joint; a per-joint array/list sets an independent limit for each
            (give the gripper a large value so grasps are not throttled). Must be > 0.
    """

    def __init__(self, max_step) -> None:
        self.max_step = np.asarray(max_step, dtype=float)
        if np.any(self.max_step <= 0):
            raise ValueError(f"max_step must be > 0, got {max_step!r}")
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, action: np.ndarray) -> np.ndarray:
        """Rate-limit one action vector against the previously emitted command."""
        a = np.asarray(action)
        dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float32
        a = a.astype(dtype, copy=False)
        if self._prev is None:  # first command: nothing to limit against, pass through
            self._prev = a.copy()
            return a
        step = self.max_step.astype(dtype, copy=False)
        out = (self._prev + np.clip(a - self._prev, -step, step)).astype(dtype, copy=False)
        self._prev = out
        return out