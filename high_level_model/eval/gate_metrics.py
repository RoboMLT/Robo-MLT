"""Deployment-shaped metrics for the System-2 completion gate.

Frame-level precision/recall on the completion head is misleading: with ``completion_window`` set to
a couple of seconds, ~30% of frames are positives, and a coarse "am I near the end?" detector already
scores well above 0.85. What actually decides whether the robot works is **when the plan pointer
switches** — a gate that fires 60 frames early on every boundary has excellent frame-level P/R and is
unusable.

So the primary metric here is closed-loop: drive :class:`System2Pipeline` frame by frame and measure
each ``advance`` against the ground-truth segment boundary it was supposed to track.

Three functions:
    - :func:`oracle_plan_from_segments` — the GT skill sequence, used as the plan (see below).
    - :func:`closed_loop_switch_report` — the primary, pointer-driven timing metric.
    - :func:`completion_ap`             — threshold-free average precision, cheap enough to log
                                          every epoch as a training curve.

**Why an oracle plan.** We are scoring the *gate*, not the planner. Without a DASHSCOPE key the
planner falls back to skill-declaration order, which is a flat list of the library's 4 skills — it
cannot express an episode that cycles grasp→place four times. Feeding the GT skill sequence isolates
the question we actually care about ("given the right plan, does the pointer switch at the right
time?") instead of confounding it with planning quality.
"""

import logging
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["closed_loop_switch_report", "completion_ap", "segment_boundaries",
           "oracle_plan_from_segments"]


def segment_boundaries(frame_to_seg: dict, ep_start: int, ep_end: int) -> List[int]:
    """Episode-local frame indices where a new GT segment begins, excluding the episode's own start.

    These are the transitions the pointer must track; the first segment needs no advance, so index 0
    is deliberately dropped.
    """
    bounds, prev = [], None
    for f in range(ep_start, ep_end):
        seg = frame_to_seg.get(f)
        key = None if seg is None else seg.start
        if prev is not None and key != prev:
            bounds.append(f - ep_start)
        prev = key
    return bounds


def oracle_plan_from_segments(frame_to_seg: dict, ep_start: int, ep_end: int,
                              library) -> tuple:
    """``(skill_ids, skill_texts)`` in GT segment order for one episode.

    Skill text is mapped back to a library id. A segment whose text is not declared in the library
    (wording drift, or a skill that has been commented out) is **registered into the library object**
    under a synthetic id rather than left as a dangling reference: the plan must keep one entry per
    segment or every later boundary silently shifts, and a plan entry that the library cannot resolve
    blows up later inside ``current_skill_text()``, far from the cause.

    Mutating the in-memory library is safe here — it is a per-eval object, never written back.
    """
    from high_level_model.planning.skill_library import Skill

    text_to_id = {library.instruction_of(sid): sid for sid in library.ids()}
    ids, texts, prev = [], [], None
    missing = []
    for f in range(ep_start, ep_end):
        seg = frame_to_seg.get(f)
        if seg is None or seg.start == prev:
            continue
        prev = seg.start
        sid = text_to_id.get(seg.skill_text)
        if sid is None:
            sid = f"__undeclared__{len(text_to_id)}"
            library.add(Skill.from_dict({"skill_id": sid,
                                         "canonical_instruction": seg.skill_text}))
            text_to_id[seg.skill_text] = sid
            missing.append(seg.skill_text)
        ids.append(sid)
        texts.append(seg.skill_text)
    if missing:
        logger.warning(
            "%d subtask text(s) in this episode are NOT declared in the skill library and were "
            "registered ad-hoc for evaluation: %s. The gate still scores them (it conditions on the "
            "text, not the id), but the deployed planner can never emit them.",
            len(set(missing)), sorted(set(missing)))
    return ids, texts


def build_eval_pipeline(library, plan: List[str], task_text: str = "eval",
                        pipeline_cls=None, planner=None, **overrides):
    """A pipeline pinned to ``plan``, with **recovery disabled**, for measuring advance timing.

    The back head is switched off (``tau_back`` set above [0, 1] and ``replan_budget=0``) so a
    recover would never fire and rewrite the plan mid-episode: recovery is a real part of
    deployment and worth measuring, but separately, and not while the quantity of interest is
    "did the pointer switch at the right frame" against the fixed oracle sequence.

    ``planner`` overrides the default offline planner — pass one carrying an ``llm_fn`` /
    ``prompt_set`` when the run *does* enable recovery and the replan should go through the LLM.
    """
    if pipeline_cls is None:
        from high_level_model.planning.system2_pipeline import System2Pipeline as pipeline_cls
    from high_level_model.planning.planner import Planner

    kwargs = dict(tau_back=1.1, replan_budget=0)
    kwargs.update(overrides)
    pipe = pipeline_cls(library, planner or Planner(library), **kwargs)
    pipe.reset(task_text)
    pipe.plan = list(plan)          # replace the planner's guess with the oracle sequence
    pipe.pointer = 0
    pipe._reset_runtime_state()
    return pipe


def completion_ap(probs: Sequence[float], labels: Sequence[float]) -> float:
    """Average precision of the completion head (area under the precision-recall curve).

    Threshold-free, so it does not merely reward matching the 0.5 operating point, and it stays
    comparable as the positive rate changes with ``completion_window``. Returns NaN when the split
    has no positives at all rather than a misleading 0.0.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(p) == 0:
        return float("nan")
    y = y > 0.5
    n_pos = float(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / n_pos)


def closed_loop_switch_report(
    step_fn: Callable[[int], Dict[str, float]],
    pipeline,
    total_frames: int,
    gt_boundaries: Sequence[int],
    match_tolerance: int = 75,
    tick_every: int = 1,
) -> dict:
    """Run the pointer over an episode and score *when* it advanced.

    ``step_fn(local_frame)`` returns ``{"back_prob", "completion_prob"}`` for that frame under the
    pipeline's CURRENT skill. The caller owns the gate forward, so this stays torch-free and is
    reusable from both the sweep and the video eval.

    Pairing: the k-th advance is matched to the k-th GT boundary. That is the right model because the
    plan is an ordered list — an early advance does not "skip ahead" to a later boundary, it
    desynchronises everything after it. An advance further than ``match_tolerance`` frames from its
    partner counts as spurious and leaves that boundary unmatched.

    Returns a JSON-serialisable dict. ``median_abs_offset`` is the headline number (lower is better;
    NaN when nothing ever matched).
    """
    # ``tick_every`` must match deployment: System 2 is stepped once every ``sampling_interval``
    # control frames (robot_inference.observe), so ``k_a`` counts TICKS, not frames. Simulating at
    # tick_every=1 makes k_a mean something ~15x shorter in wall-clock than the same number on the
    # robot — the single easiest way to transplant a bogus threshold onto the real system.
    switch_frames: List[int] = []
    for local_f in range(0, total_frames, max(1, int(tick_every))):
        if pipeline.is_done():
            break
        sig = step_fn(local_f)
        info = pipeline.step(back_prob=sig["back_prob"], completion_prob=sig["completion_prob"])
        # "done" fires when the FINAL skill completes; the episode has no boundary there, so it is
        # reported via `completed` rather than counted as a switch.
        if info.get("action") == "advance":
            switch_frames.append(local_f)

    gt = list(gt_boundaries)
    matched: List[int] = []
    n_spurious = 0
    for k, sf in enumerate(switch_frames):
        if k >= len(gt) or abs(sf - gt[k]) > match_tolerance:
            n_spurious += 1
        else:
            matched.append(sf - gt[k])
    n_missed = len(gt) - len(matched)

    abs_off = [abs(o) for o in matched]
    return {
        "n_gt_boundaries": len(gt),
        "n_switches": len(switch_frames),
        "switch_frames": switch_frames,
        "gt_boundaries": gt,
        "offsets": matched,                                                   # + = late, - = early
        "median_abs_offset": float(np.median(abs_off)) if abs_off else float("nan"),
        "mean_offset": float(np.mean(matched)) if matched else float("nan"),
        "n_missed": int(n_missed),
        "n_spurious": int(n_spurious),
        "completed": bool(pipeline.is_done()),
        "match_tolerance": int(match_tolerance),
    }
