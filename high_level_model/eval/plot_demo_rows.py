"""One task = one row: stack the System-2 closed-loop demos into a single figure (PNG/PDF/PPTX).

``plot_inference_demo`` (TubeSort, *advance* story) and ``plot_back_recovery_demo`` (BloodGas,
*back/recovery* story) each render their own figure. For a paper/slide panel the two belong side by
side, in one layout. This module is that renderer: it consumes the ``.data.pkl`` closed-loop caches
those two scripts already write, normalises their two different schemas, and draws
**one row per task** —

    cam_top film strip  →  System-2 pointer bar  →  gate-signal curve panel

— then stacks the rows and emits a combined PNG/PDF plus an **editable PowerPoint block diagram**.
No GPU, no dataset, no re-inference: everything comes out of the caches.

What this renderer does that the per-task scripts do not:

* **Every row starts at t = 0.** A cropped row is re-zeroed to its own first frame, and the leading
  ``--warmup_frames`` are shaded as the gate's history warm-up window. On a cropped row that band is
  a *nominal* marker of the window length, not a measurement — the gate was already warm there.
* **An explicit ``Replanning`` block.** When a row carries a recover, the interval from the recover
  to the end of the interference window is carved out of the head of the next pointer segment and
  drawn as its own block: System-2 has dropped the plan and is waiting for a new subtask list, so
  the robot is idle. Both gate signals read exactly 0 across it.
* **No valid-switch window inside the replanning interval.** A GT completion window that overlaps
  the replanning interval is clipped away: there is nothing to switch *to* while the plan is being
  rebuilt.
* **Elisions.** ``--elide`` cuts a slice out of the middle of an over-long subtask so it stops
  squeezing every other block, marking the cut with the usual axis-break slashes. Time stays linear
  either side of the break, and nothing is drawn from inside the cut (see :class:`TimeMap`).
* **No prose in the plot area.** No callouts, no arrows, no in-panel captions — every label is axis
  furniture in a gutter or a legend entry, leaving the panels clear to annotate by hand afterwards.

Run from the repo root::

    python -m high_level_model.eval.plot_demo_rows \
        --row bloodgas:outputs/completion_gate_bloodgas_merged/back_fig_ep0001.data.pkl:7:10 \
        --row tubesort:outputs/sweep_20260720/paper_fig_ep189.data.pkl:0:9 \
        --fetch bloodgas:/path/to/BloodGasAnalysis_20260726_merged:1:cam_top \
        --replan_frame 1785 --elide bloodgas:2895:6.0 \
        --tau_back 0.8 --tau_done 0.6 \
        --out outputs/completion_gate_bloodgas_merged/demo_rows

``--row <name>:<pkl>:<seg_start>:<seg_count>``; pass one ``--row`` for a single-row figure. Rows fill
the width equally by default; ``--shared_scale`` draws them at a common seconds-per-inch instead.
"""

import argparse
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, ConnectionPatch

from high_level_model.eval.plot_inference_demo import SHORT, _contiguous_runs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

C_BACK = "#e87ba4"      # back probability
C_COMP = "#2a78d6"      # completion probability
C_TAU_BACK = "#c2185b"
C_TAU_DONE = "#eb6834"
C_WINDOW = "#1baf7a"    # GT valid-switch (completion) window
C_WARM = "#8a8a86"      # history warm-up band
C_INK = "#111111"
C_MUTE = "#4a4a47"
C_PLACEHOLDER = "#e9e9e5"

# Slot 7 is re-stepped from ``plot_inference_demo``'s #17a2b8 (teal) to #b5179e (magenta). In this
# figure "Discard sample" abuts "Grasp blood sample" (#2a78d6) at the end of row (a), and teal↔blue
# is ΔE 13.8 on the validator's normal-vision floor — a hard FAIL that secondary encoding does not
# excuse. Magenta takes the worst adjacent pair to 19.6. Two residuals are accepted and are
# pre-existing properties of the eight-hue set, not of this change:
#   · #1baf7a↔#e87ba4 deutan ΔE 6.1 — inside the 6–8 floor band, legal here because identity is
#     never colour-alone: every band carries its name in-band and every film frame is titled.
#   · #8a63d2↔#2a78d6 under `--pairs all` — those two never abut in either row.
ROW_PALETTE = ["#2a78d6", "#eda100", "#e87ba4", "#1baf7a",
               "#8a63d2", "#eb6834", "#b5179e", "#b3541e"]

# Replanning is a *system state*, not a skill, so it gets a reserved status colour rather than a slot
# in the categorical palette — a categorical slot would imply it is one more subtask in the plan.
# It sits just below the categorical lightness band (OKLCh L 0.429 vs the 0.43–0.77 peer band) on
# purpose: that is what makes it read as "not one of the skills". Checked as a lone status colour it
# passes contrast, so the white bold label on it is legible.
C_REPLAN = "#8f1d2f"

REPLAN_KEY = "__replanning__"
REPLAN_LABEL = "Replanning"

# Display names for the figure. Keyed by row name because "grasp the routine blood tube." means a
# blood-gas sample in one capture and a routine tube to be sorted in the other.
TASK_LABELS: Dict[str, Dict[str, str]] = {
    "bloodgas": {
        "grasp the routine blood tube.":                                 "Grasp blood sample",
        "remove the sample tube cap.":                                   "Remove cap",
        "discard the tube cap into the waste bin.":                      "Discard cap",
        "dock the blood gas sample tube to the blood gas analyzer.":     "Dock analyzer",
        "insert the sample tube into the analyzer probe.":               "Insert probe",
        "withdraw the blood gas sample tube after testing is complete.": "Withdraw tube",
        "discard the tested blood sample into the waste bin.":           "Discard sample",
        "return the robotic arm to the home position.":                  "Home",
    },
    "tubesort": {
        "grasp the routine blood tube.":        "Grasp routine tube",
        "grasp the sodium citrate tube.":       "Grasp sodium tube",
        "drop the tube into the orange plate.": "→ Orange plate",
        "place to the blue plate.":             "→ Blue plate",
        "return to the home position.":         "Home",
    },
}

ROW_CAPTIONS: Dict[str, str] = {
    "bloodgas": "Blood-gas analysis — human interference, replanning and recovery",
    "tubesort": "Blood-tube sorting — nominal subtask advancement",
}

INTERVENTION_TEXT = ("Manual Intervention: the sample was removed by an operator during the "
                     "robotic inspection cycle.\nSystem-2 replans and resumes with the next sample.")


def _label_for(name: str, text: str) -> str:
    return TASK_LABELS.get(name, {}).get(text) or SHORT.get(text, text)


def _wrap(label: str, width: int = 14) -> str:
    """Soft-wrap a film-strip title so a long name does not run into its neighbours' slots."""
    if len(label) <= width:
        return label
    out, line = [], ""
    for word in label.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return "\n".join(out)


# ─────────────────────────── cache normalisation ────────────────────────────────
def load_row(name: str, pkl: str, seg_start: int, seg_count: int, args) -> dict:
    """Normalise either closed-loop cache schema into one row spec.

    ``plot_back_recovery_demo`` writes ``view_comp``/``view_back``/``ptr_texts``/``advance_frames``
    and a *dict* of key frames; ``plot_inference_demo`` writes ``comp_probs``/``ptr_skills`` and a
    *list*. Everything downstream sees the same keys.
    """
    with open(pkl, "rb") as f:
        d = pickle.load(f)

    total = int(d["total"])
    fps = int(d["fps"])
    ptr_change = list(d["ptr_change"])
    ptr_texts = list(d.get("ptr_texts") or d["ptr_skills"])
    n = len(ptr_change)
    if seg_start < 0 or seg_start + seg_count > n:
        raise ValueError(f"[{name}] segments [{seg_start}, {seg_start + seg_count}) "
                         f"out of range (cache has {n})")

    advance = list(d.get("advance_frames") or ptr_change[1:])
    recover = list(d.get("recover_frames") or [])

    # Both rows are rebuilt through one code path, starting from the *raw per-frame* gate output
    # that each cache stores, so the plotted curve is the pipeline's own signal reconstructed at the
    # pipeline's own rate (see :func:`_tick_filter`). The cached ``filt_*``/``view_*`` arrays are not
    # reused: both were filtered once per frame rather than once per tick, which puts the curve above
    # τ_done seconds before the switch it caused. Rebuilding is also what lets ``--tick_stride``
    # differ per row — the switch frames themselves always stay where the recorded closed loop put
    # them, since re-deciding those would need the checkpoint and a GPU.
    warm = int(d.get("history_span") or args.warmup_frames)
    stride = int(args.tick_stride_by.get(name, args.tick_stride))
    switches = sorted(set(advance) | set(recover))
    raw_comp = list(d["raw_comp"] if "raw_comp" in d else d["comp_probs"])

    # The two source scripts advance the output filter at different rates, and the plotted curve has
    # to use whichever rate the recorded run actually thresholded — otherwise it cannot explain its
    # own switch frames. ``plot_back_recovery_demo`` calls ``comp_filt.update`` once per frame
    # (outside its tick guard) and so is reconstructed per-frame; ``plot_inference_demo`` keeps the
    # filter inside ``System2Pipeline.step``, which fires once per tick. Deployment
    # (``System2Controller``) is tick-rate, so the tubesort row is the deployment-faithful one and
    # the bloodgas row is reproducing a script that filters faster than the robot does.
    rate = args.filter_rate_by.get(name, "auto")
    per_frame = _detect_per_frame(d, stride) if rate == "auto" else (rate == "per_frame")

    # ``--curve_source`` picks how much temporal resolution the drawn curve keeps. It changes only
    # what is plotted; the switch frames, the latch and every shaded window are untouched.
    #   tick      — one step per System-2 decision. The only setting where the curve crosses tau on
    #               the very sample that fires the switch, but visibly coarse (one step per 0.6 s).
    #   per_frame — the pipeline's own median+EMA advanced every frame. Smooth and fine-grained (this
    #               is what plot_inference_demo draws), at the cost of reaching tau before the tick
    #               that actually acts on it.
    #   raw       — the unfiltered per-frame sigmoid. Finest, and honestly spiky: adjacent frames
    #               differ by up to ~0.9.
    src = args.curve_source
    if src == "auto":
        src = "per_frame" if per_frame else "tick"
    logger.info("[%s] curve source = %s (run filtered %s, tick_stride=%d)", name, src,
                "per frame" if per_frame else "once per tick", stride)

    def _prepare(raw, median_window):
        if src == "raw":
            return np.asarray(raw, dtype=float)
        return _tick_filter(raw, switches, warm, tick_stride=stride,
                            median_window=median_window, per_frame=(src == "per_frame"))

    comp = _controller_view(
        _prepare(raw_comp, 3), switches, warm, tau=float(d.get("tau") or args.tau_done),
        k=int(d.get("k_a") or args.k_a), cooldown=args.cooldown, tick_stride=stride)

    back = None
    if "raw_back" in d:
        # The back filter runs with no median window (median_window=1) in the pipeline, and its latch
        # is set only by a recover: an advance consumes a completion plateau, not an interference
        # event.
        back = _controller_view(
            _prepare(list(d["raw_back"]), 1), recover, warm, tau=args.tau_back, k=args.k_b,
            cooldown=args.cooldown, tick_stride=stride)
    elif "view_back" in d:
        back = np.asarray(d["view_back"], dtype=float)

    # Key frames: dict keyed by pointer-segment start, or list in pointer order.
    kf = d["key_frames"]
    def _key_frame(i: int, start: int):
        if isinstance(kf, dict):
            return kf.get(start)
        return kf[i] if 0 <= i < len(kf) else None

    uniq: List[str] = []
    for txt in (d.get("plan_texts") or ptr_texts):
        if txt not in uniq:
            uniq.append(txt)
    colors = {txt: ROW_PALETTE[i % len(ROW_PALETTE)] for i, txt in enumerate(uniq)}

    bounds = ptr_change + [total]
    blocks = []
    for i in range(seg_start, seg_start + seg_count):
        blocks.append(dict(index=i, start=bounds[i], end=bounds[i + 1], text=ptr_texts[i],
                           label=_label_for(name, ptr_texts[i]),
                           color=colors.get(ptr_texts[i], ROW_PALETTE[0]),
                           key_frame=_key_frame(i, bounds[i]), frame_at=bounds[i],
                           cached_at=bounds[i]))

    lo, hi = blocks[0]["start"], blocks[-1]["end"]
    row = dict(name=name, pkl=pkl, blocks=blocks, lo=lo, hi=hi, total=total, fps=fps,
               comp=comp, back=back, advance=advance, recover=recover,
               comp_gt=_runs(d.get("comp_gt_locals")), back_gt=_runs(d.get("back_gt_locals")),
               warmup=int(d.get("history_span") or args.warmup_frames),
               episode=d.get("episode"), dataset_name=d.get("dataset_name") or "",
               caption=ROW_CAPTIONS.get(name, name), replan=None)
    _insert_replanning(row, args)
    return row


def _detect_per_frame(d: dict, tick_stride: int) -> bool:
    """Did the recorded run advance its output filter every frame, or once per tick?

    Read it off the cached signal rather than off which keys are present: both cadences write the
    same ``filt_comp`` field, so keying on presence silently mislabels a tick-rate run. A tick-rate
    filter is held between ticks, so its value can only change at multiples of ``tick_stride``.
    ``plot_back_recovery_demo`` records ``filter_rate`` directly from 2026-07-26 on; that is
    authoritative when present.
    """
    if "filter_rate" in d:
        return str(d["filter_rate"]) == "per_frame"
    if "filt_comp" not in d:
        return False                      # plot_inference_demo caches: filter lives in pipe.step
    fc = np.asarray(d["filt_comp"], dtype=float)
    changed = np.nonzero(np.abs(np.diff(fc)) > 1e-12)[0] + 1
    if len(changed) == 0:
        return False
    off_tick = int((changed % max(1, tick_stride) != 0).sum())
    return off_tick > 0.05 * len(changed)


def _tick_filter(raw, switches: List[int], warmup: int, *, tick_stride: int,
                 median_window: int = 3, ema_alpha: float = 0.5,
                 per_frame: bool = False) -> np.ndarray:
    """Reproduce ``System2Pipeline``'s output filter *at the rate the pipeline actually runs it*.

    This is the difference between a curve you can read the switches off and one you cannot.
    ``System2Pipeline.step`` — and with it ``_comp_filter.update`` — is called once every
    ``tick_stride`` control frames, not once per frame. Feeding the same filter every frame (which is
    what a naive replay does) advances its EMA ~``tick_stride``x faster, so the plotted curve reaches
    τ_done whole seconds before the value the controller is thresholding does. On the TubeSort
    episode that put the curve above τ for a median of 1.7 s — and up to 7.2 s — before the switch
    it supposedly caused, which reads as a badly-tuned controller. Sampled at the true tick rate the
    same run crosses τ **0.04 s** before firing.

    Between ticks the controller's view simply does not change, so the value is held: the curve is a
    staircase, one step per System-2 decision, which is the honest picture of a 1 Hz reasoner
    supervising a 20 Hz loop. ``per_frame=True`` restores the old behaviour for comparison.

    The reset ordering matters and is easy to get backwards. The recorded loop does
    *update, then decide, then reset*: the value that fires a switch is the one the filter held
    **before** it was cleared, so the reset only takes effect from the next tick. Resetting before
    the update instead throws away one tick of EMA at every switch, which flattens every peak — on
    the bloodgas episode it capped the curve at ~0.5 against a τ_done of 0.6, i.e. switches with no
    visible cause.
    """
    from high_level_model.planning.signal_filters import ProgressSignalFilter
    f = ProgressSignalFilter(median_window=median_window, ema_alpha=ema_alpha, monotone=False)
    resets = set(int(s) for s in switches)
    out, held, pending = np.empty(len(raw), dtype=float), 0.0, False
    for i, v in enumerate(raw):
        if per_frame or (i >= warmup and i % max(1, tick_stride) == 0):
            if pending:
                f.reset(); pending = False
            held = f.update(float(v))
        if i in resets:
            pending = True
        out[i] = held
    return out


def _controller_view(comp: np.ndarray, switches: List[int], warmup: int, *, tau: float,
                     k: int, cooldown: int, tick_stride: int) -> np.ndarray:
    """Latch the completion signal to 0 once the plateau that caused a switch has been acted on.

    The latch is released only after the signal has stayed sub-τ for a full debounce window *and*
    the cooldown has expired — releasing on the first sub-threshold sample instead lets an
    already-consumed plateau pop back up moments later, a "second" event the controller never sees.

    It engages from the **next tick**, not from the switch frame itself. The sample that crosses τ
    and the switch it triggers land on the same tick, so latching immediately would blank the very
    value that caused the switch: with a tick-rate filter the curve could then never be seen crossing
    its own threshold, and every switch would look unexplained. Holding one tick shows the causing
    peak, then consumes it.
    """
    sw = set(int(s) for s in switches)
    out = np.asarray(comp, dtype=float).copy()
    latched, below, cd, pending = False, 0, 0, False
    for f in range(len(out)):
        if f >= warmup and f % max(1, tick_stride) == 0:
            if pending:
                latched, below, cd, pending = True, 0, cooldown, False
            elif latched:
                if cd > 0:
                    cd -= 1
                below = below + 1 if out[f] < tau else 0
                if below >= max(2, k) and cd == 0:
                    latched = False
        if f in sw:
            pending = True
        if latched:
            out[f] = 0.0
    return out


class TimeMap:
    """Frame index → display seconds for one row, with elided frame ranges collapsed out.

    A subtask that runs much longer than its neighbours squeezes every other block on the timeline.
    Rather than rescale it (which would silently break the axis), a slice is cut from the middle and
    the cut is marked on the axis with the usual break slashes: time stays linear on both sides of
    the break, and nothing is drawn from inside the cut.
    """

    def __init__(self, lo: int, hi: int, fps: int, cuts=()):
        self.lo, self.hi, self.fps = int(lo), int(hi), int(fps)
        self.cuts = sorted((int(a), int(b)) for a, b in cuts if b > a)

    def t(self, frame) -> float:
        f = min(max(int(frame), self.lo), self.hi)
        removed = sum(min(max(f - a, 0), b - a) for a, b in self.cuts)
        return (f - self.lo - removed) / self.fps

    def t_arr(self, frames: np.ndarray) -> np.ndarray:
        f = np.clip(np.asarray(frames, dtype=float), self.lo, self.hi)
        removed = np.zeros_like(f)
        for a, b in self.cuts:
            removed += np.clip(f - a, 0, b - a)
        return (f - self.lo - removed) / self.fps

    @property
    def span(self) -> float:
        return self.t(self.hi)

    def pieces(self, a: int, b: int) -> List[Tuple[float, float]]:
        """Visible display-second intervals of the frame range [a, b)."""
        a, b = max(int(a), self.lo), min(int(b), self.hi)
        if b <= a:
            return []
        out, cur = [], a
        for c0, c1 in self.cuts:
            if c1 <= cur or c0 >= b:
                continue
            if c0 > cur:
                out.append((self.t(cur), self.t(c0)))
            cur = max(cur, c1)
        if cur < b:
            out.append((self.t(cur), self.t(b)))
        return [(x, y) for x, y in out if y > x]

    def hidden(self, frames: np.ndarray) -> np.ndarray:
        m = np.zeros(len(frames), dtype=bool)
        for a, b in self.cuts:
            m |= (frames >= a) & (frames < b)
        return m

    @property
    def breaks(self) -> List[float]:
        return [self.t(a) for a, _ in self.cuts if self.lo < a < self.hi]


def apply_elisions(rows: List[dict], specs) -> None:
    """Attach a :class:`TimeMap` to every row, honouring ``--elide <row>:<block_start>:<seconds>``."""
    wanted: Dict[Tuple[str, int], float] = {(n, f): s for n, f, s in (specs or [])}
    for row in rows:
        cuts = []
        for b in row["blocks"]:
            secs = wanted.pop((row["name"], b["start"]), None)
            if secs is None:
                continue
            drop = int(round(secs * row["fps"]))
            keep = (b["end"] - b["start"]) - drop
            if keep < row["fps"]:                     # never collapse a block below ~1 s
                logger.warning("[%s] refusing to cut %.1f s from f%d–%d (only %.1f s long)",
                               row["name"], secs, b["start"], b["end"],
                               (b["end"] - b["start"]) / row["fps"])
                continue
            mid = (b["start"] + b["end"]) // 2
            cuts.append((mid - drop // 2, mid - drop // 2 + drop))
            logger.info("[%s] eliding %.1f s from the middle of '%s' (f%d–%d → %.1f s shown)",
                        row["name"], secs, b["label"], b["start"], b["end"], keep / row["fps"])
        row["tmap"] = TimeMap(row["lo"], row["hi"], row["fps"], cuts)
    for (name, frame) in wanted:
        logger.warning("--elide %s:%d did not match any block start", name, frame)


def _runs(locals_) -> List[Tuple[int, int]]:
    """GT frame list → inclusive (start, end) runs. Tolerates None / numpy input."""
    if locals_ is None:
        return []
    seq = [int(x) for x in np.asarray(locals_).ravel().tolist()]
    return _contiguous_runs(sorted(seq))


def _insert_replanning(row: dict, args) -> None:
    """Carve the replanning interval out of the head of the segment that follows the recover.

    The interval is [recover, end-of-interference-window). No time is inserted — the following
    block simply starts later — so every other block keeps its recorded timing.
    """
    if args.no_replan or not row["recover"]:
        return
    rec = next((f for f in row["recover"] if row["lo"] <= f < row["hi"]), None)
    if rec is None:
        return

    if args.replan_end == "auto":
        end = next((b + 1 for a, b in row["back_gt"] if a <= rec <= b), None)
        if end is None:
            end = rec + int(round(args.replan_fallback_s * row["fps"]))
    else:
        end = int(args.replan_end)

    idx = next((k for k, b in enumerate(row["blocks"]) if b["start"] == rec), None)
    if idx is None:
        logger.warning("[%s] recover at f%d is not a pointer-segment start; no Replanning block",
                       row["name"], rec)
        return
    end = min(end, row["blocks"][idx]["end"])

    # The Replanning thumbnail should show *why* the plan was dropped. The physical intervention
    # precedes both the annotated back window and the gate's detection of it, so the most legible
    # frame usually sits outside the block's own interval — hence `--replan_frame`, and hence the
    # separate strip caption, which names the depicted event and its own frame number rather than
    # implying the photo was taken inside the block.
    hole = next(((a, b) for a, b in row["back_gt"] if a <= rec <= b), (rec, end))
    shot = (hole[0] + hole[1]) // 2 if args.replan_frame == "auto" else int(args.replan_frame)
    nxt = row["blocks"][idx]
    replan = dict(index=-1, start=rec, end=end, text=REPLAN_KEY, label=REPLAN_LABEL,
                  strip_label=args.replan_strip_label, color=C_REPLAN, key_frame=nxt["key_frame"],
                  is_replan=True, frame_at=shot, cached_at=nxt["cached_at"])
    nxt["start"] = end
    nxt["key_frame"] = None                       # the cached frame belongs to the recover instant
    nxt["frame_at"] = nxt["cached_at"] = end
    if args.post_recover_label:
        nxt["label"] = args.post_recover_label
    row["blocks"].insert(idx, replan)
    row["replan"] = (rec, end)
    logger.info("[%s] Replanning block f%d–%d (%.1f s); '%s' now starts at f%d",
                row["name"], rec, end, (end - rec) / row["fps"], nxt["label"], end)


def fetch_frames(rows: List[dict], specs, cache_path: str) -> None:
    """Decode the thumbnails the caches do not hold (the Replanning and post-recover frames).

    Only frames whose ``frame_at`` differs from the frame the cache actually stored are decoded, so
    a normal run touches the dataset zero times. Results are memoised to ``cache_path`` — the second
    run of the same figure needs no dataset at all.
    """
    wanted = {(r["name"], b["frame_at"]) for r in rows for b in r["blocks"]
              if b["key_frame"] is None or b["frame_at"] != b["cached_at"]}
    if not wanted:
        return

    store: Dict[Tuple[str, int], np.ndarray] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            store = pickle.load(f)

    todo = {k for k in wanted if k not in store}
    by_row = {name: (repo, ep, cam) for name, repo, ep, cam in (specs or [])}
    if todo and by_row:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torchvision.transforms import v2 as T
        from high_level_model.data.high_level_dataset import _get_ep_bounds
        for name in sorted({n for n, _ in todo}):
            if name not in by_row:
                continue
            repo, ep, cam = by_row[name]
            ds = LeRobotDataset(repo_id=repo, tolerance_s=1e-2,
                                image_transforms=T.Compose([T.ToDtype(torch.float32, scale=True)]))
            ep_start, _ = _get_ep_bounds(ds.meta, ep)
            for n, f in sorted(k for k in todo if k[0] == name):
                t = ds[ep_start + f][f"observation.images.{cam}"]
                store[(n, f)] = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                logger.info("[%s] decoded thumbnail at f%d from %s", n, f, repo)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(store, f)

    for r in rows:
        for b in r["blocks"]:
            img = store.get((r["name"], b["frame_at"]))
            if img is not None:
                b["key_frame"] = img
                b["cached_at"] = b["frame_at"]
    missing = [(r["name"], b["frame_at"]) for r in rows for b in r["blocks"]
               if b["key_frame"] is None]
    if missing:
        logger.warning("no thumbnail for %s — pass --fetch <row>:<repo_id>:<episode>:<camera>",
                       missing)


def _clip_out(runs: List[Tuple[int, int]], hole: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Remove ``hole`` from each run, keeping whatever falls outside it."""
    if hole is None:
        return runs
    h0, h1 = hole
    out = []
    for a, b in runs:
        if b < h0 or a >= h1:
            out.append((a, b))
            continue
        if a < h0:
            out.append((a, h0 - 1))
        if b >= h1:
            out.append((h1, b))
    return [(a, b) for a, b in out if b > a]


# ─────────────────────────── shared curve drawing ───────────────────────────────
def _draw_signals(ax, row: dict, args, *, decorate: bool):
    """Curves + thresholds + GT shading for one row.

    Nothing inside the panel is prose: no callouts, no arrows, no in-plot captions. Every label is
    either axis furniture in the gutters or lives in the legend, so the plot area stays free for
    annotation by hand afterwards.
    """
    tm: TimeMap = row["tmap"]
    lo, hi, span_s = row["lo"], row["hi"], tm.span
    lw = args.lw_scale

    # A valid-switch window is clipped out of two intervals where no switch can happen: the
    # replanning interval (the plan it would advance along no longer exists) and the history warm-up
    # (the gate has not emitted anything yet, so there is nothing to threshold).
    warm = (lo, lo + row["warmup"])
    for a, b in _clip_out(_clip_out(row["comp_gt"], row["replan"]), warm):
        for x0, x1 in tm.pieces(a, b):
            ax.axvspan(x0, x1, color=C_WINDOW, alpha=0.16, lw=0, zorder=0)
    # The interference window overlaps the valid-switch window near a boundary; two stacked alphas
    # blend into a muddy third colour that reads as its own category. So the interference span gets a
    # faint tint plus a solid cap bar along the top of the panel — unambiguous under overlap.
    for a, b in row["back_gt"]:
        for x0, x1 in tm.pieces(a, b):
            ax.axvspan(x0, x1, color=C_BACK, alpha=0.10, lw=0, zorder=0)
            # y 1.005–1.055 is inside ylim, so default clipping keeps the bar in the panel. Do NOT
            # set clip_on=False: a run outside this row's x-range would then draw far off-axes and
            # bbox_inches="tight" would expand the whole canvas around it.
            ax.add_patch(plt.Rectangle((x0, 1.005), x1 - x0, 0.05, color=C_BACK, lw=0, zorder=6))

    for x0, x1 in tm.pieces(lo, lo + row["warmup"]):
        ax.axvspan(x0, x1, color=C_WARM, alpha=0.16, lw=0, zorder=0)

    ax.axhline(args.tau_done, ls="--", lw=1.0 * lw, color=C_TAU_DONE, zorder=2)
    if row["back"] is not None:
        ax.axhline(args.tau_back, ls="--", lw=1.0 * lw, color=C_TAU_BACK, zorder=2)

    # Frames inside an elision are set to NaN rather than dropped: that breaks the polyline exactly
    # at the cut instead of drawing a straight bridge across it.
    idx = np.arange(lo, hi)
    x = tm.t_arr(idx)
    # Blank inside an elision (breaks the polyline at the cut instead of bridging it) and inside the
    # warm-up window: the gate has no history to run on there, so it emits nothing and the band must
    # not carry a curve.
    blank = tm.hidden(idx) | (idx < lo + row["warmup"])
    comp = row["comp"][lo:hi].astype(float).copy(); comp[blank] = np.nan
    ax.plot(x, comp, lw=1.2 * lw, color=C_COMP, zorder=4, solid_joinstyle="round")
    if row["back"] is not None:
        back = row["back"][lo:hi].astype(float).copy(); back[blank] = np.nan
        ax.plot(x, back, lw=1.5 * lw, color=C_BACK, zorder=5, solid_joinstyle="round")

    for f in row["advance"]:
        if lo < f < hi and not tm.hidden(np.array([f]))[0]:
            ax.axvline(tm.t(f), color=C_INK, ls=(0, (4, 2)), lw=0.8 * lw, alpha=0.75, zorder=3)
    for f in row["recover"]:
        if lo <= f < hi and not tm.hidden(np.array([f]))[0]:
            ax.axvline(tm.t(f), color=C_TAU_BACK, lw=1.7 * lw, zorder=7)

    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(0, span_s)
    _draw_axis_breaks(ax, tm, span_s, decorate=decorate)
    if decorate:
        ax.set_ylabel("gate\nprobability", fontsize=10)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(axis="y", color="#e8e8e5", lw=0.6)
        ax.tick_params(labelsize=9.5)
        # τ labels sit in the right-hand gutter, *outside* the axes: inside the panel they would be
        # in-plot text, which this figure deliberately has none of.
        ax.text(span_s + 0.008 * span_s, args.tau_done, f"τ_done = {args.tau_done:g}",
                ha="left", va="center", fontsize=9, color=C_TAU_DONE, zorder=7, clip_on=False)
        if row["back"] is not None:
            ax.text(span_s + 0.008 * span_s, args.tau_back, f"τ_back = {args.tau_back:g}",
                    ha="left", va="center", fontsize=9, color=C_TAU_BACK, zorder=7, clip_on=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)


def _draw_axis_breaks(ax, tm: TimeMap, span_s: float, *, decorate: bool) -> None:
    """Standard break slashes on the bottom spine wherever a slice of time was elided."""
    if not tm.breaks:
        return
    trans = ax.get_xaxis_transform()          # x in data units, y in axes fraction
    w = 0.006 * span_s
    for xb in tm.breaks:
        ax.add_patch(plt.Rectangle((xb - 1.6 * w, -0.035), 3.2 * w, 0.07, transform=trans,
                                   facecolor="white", edgecolor="none", zorder=8, clip_on=False))
        if decorate:
            for off in (-0.9 * w, 0.9 * w):
                ax.plot([xb + off - w, xb + off + w], [-0.038, 0.038], transform=trans,
                        color=C_INK, lw=0.9, zorder=9, clip_on=False, solid_capstyle="butt")


# ───────────────────────── matplotlib PNG / PDF ─────────────────────────────────
def render_png(rows: List[dict], args, out_stem: str) -> Tuple[str, str]:
    n_rows = len(rows)
    max_blocks = max(len(r["blocks"]) for r in rows)
    span_max = max(r["hi"] - r["lo"] for r in rows)

    fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    # `right` leaves a gutter for the τ labels, which are drawn outside the curve panels.
    outer = fig.add_gridspec(n_rows, 1, hspace=0.46, left=0.050, right=0.930,
                             top=0.868 if args.title else 0.930, bottom=0.090)

    legend_ax = None
    for r, row in enumerate(rows):
        tm: TimeMap = row["tmap"]
        lo, hi, span_s = row["lo"], row["hi"], tm.span
        frac = (hi - lo) / span_max if args.shared_scale else 1.0

        blk = outer[r].subgridspec(3, 1, height_ratios=[2.05, 0.32, 3.10], hspace=0.06)
        strip_gs = blk[0].subgridspec(1, max_blocks, wspace=0.045)

        ax_curve = fig.add_subplot(blk[2])
        ax_track = fig.add_subplot(blk[1])
        # Shared seconds-per-inch across rows: the longest row fills the width, the others end short.
        for ax in (ax_curve, ax_track):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width * frac, pos.height])

        # ── film strip: one cam_top thumbnail per block, evenly spaced ──
        frame_axes = []
        for c, b in enumerate(row["blocks"]):
            axf = fig.add_subplot(strip_gs[0, c])
            if b["key_frame"] is not None:
                axf.imshow(b["key_frame"])
            else:
                axf.set_facecolor(C_PLACEHOLDER)
                axf.text(0.5, 0.5, "no cached\nframe", ha="center", va="center",
                         fontsize=8, color=C_MUTE, transform=axf.transAxes)
            axf.set_xticks([]); axf.set_yticks([])
            for sp in axf.spines.values():
                sp.set_edgecolor(b["color"]); sp.set_linewidth(2.8)
            tag = " ⟲" if b.get("is_replan") else ""
            axf.set_title(f"{_wrap(b.get('strip_label') or b['label'])}{tag}\nf{b['frame_at']}",
                          fontsize=9.0, color=C_INK, pad=3, linespacing=1.15)
            frame_axes.append(axf)
        for c in range(len(row["blocks"]), max_blocks):
            fig.add_subplot(strip_gs[0, c]).set_axis_off()

        # ── System-2 pointer bar ──
        # Inches of track per second, so the label fit test is in real typographic units rather than
        # a magic constant that silently breaks when the figure is resized.
        in_per_s = ax_track.get_position().width * fig.get_figwidth() / span_s
        for b in row["blocks"]:
            a0, a1 = tm.t(b["start"]), tm.t(b["end"])
            ax_track.axvspan(a0, a1, color=b["color"], alpha=0.95, lw=0)
            # Step the size down until the whole word fits; Calibri-ish bold averages ~0.58 em per
            # character. A band too narrow even at 7 pt is left blank rather than clipped — its
            # film-strip title still names it.
            for size in (9.5, 8.5, 7.5, 7.0):
                if (a1 - a0) * in_per_s > 0.58 * (size / 72.0) * len(b["label"]) + 0.06:
                    ax_track.text((a0 + a1) / 2, 0.5, b["label"], ha="center", va="center",
                                  fontsize=size, color="white", weight="bold")
                    break
            ax_track.axvline(a1, color="white", lw=1.8, zorder=3)
        # Break notation on the bar too — and it must not look like the plain white block separator,
        # or a cut inside one subtask reads as a boundary between two.
        bw = 0.005 * span_s
        for xb in tm.breaks:
            ax_track.axvspan(xb - 1.5 * bw, xb + 1.5 * bw, color="white", lw=0, zorder=4)
            for off in (-0.8 * bw, 0.8 * bw):
                ax_track.plot([xb + off - bw, xb + off + bw], [-0.15, 1.15], color=C_INK,
                              lw=0.9, zorder=5, clip_on=False, solid_capstyle="butt")
        ax_track.set_yticks([]); ax_track.set_ylim(0, 1); ax_track.set_xlim(0, span_s)
        ax_track.set_ylabel("subtask\n(System-2)", fontsize=9.5, rotation=0, ha="right",
                            va="center")
        for sp in ax_track.spines.values():
            sp.set_visible(False)
        plt.setp(ax_track.get_xticklabels(), visible=False)

        # ── curve panel ──
        _draw_signals(ax_curve, row, args, decorate=True)
        ax_curve.set_xlabel("time (s)", fontsize=10.5)

        # The row caption goes above the film strip via fig.text, not as an axes title: an axes title
        # on either the track or the curve lands inside the strip, where the opaque thumbnail axes
        # paint over it.
        tag = f"({'abcd'[r]})  {row['caption']}"
        if row.get("episode") is not None:
            tag += f"   ·   episode {row['episode']}"
            if row.get("dataset_name"):
                tag += f" ({row['dataset_name']})"
        strip_top = max(a.get_position().y1 for a in frame_axes)
        # The offset is derived from the tallest film-strip title in this row rather than fixed: the
        # titles wrap to one, two or three lines depending on the skill names, and a constant gap
        # lands the caption inside them for the three-line case.
        n_lines = max(a.get_title().count("\n") + 1 for a in frame_axes)
        pad = (n_lines * 9.0 * 1.18 + 9.0) / 72.0 / fig.get_figheight()
        fig.text(0.050, strip_top + pad, tag, fontsize=12.5, color=C_MUTE, ha="left", va="bottom")

        for c, b in enumerate(row["blocks"]):
            fig.add_artist(ConnectionPatch(
                xyA=(0.5, 0.0), coordsA=frame_axes[c].transAxes,
                xyB=((tm.t(b["start"]) + tm.t(b["end"])) / 2, 1.0), coordsB=ax_track.transData,
                color="#b9b9b4", lw=0.8, zorder=0))
        legend_ax = ax_curve

    last_frac = ((rows[-1]["hi"] - rows[-1]["lo"]) / span_max) if args.shared_scale else 1.0
    legend_ax.legend(handles=_legend_handles(rows, args), ncol=args.legend_cols, fontsize=9.2,
                     loc="upper center", frameon=False, columnspacing=1.3, handlelength=1.7,
                     bbox_to_anchor=(0.5 / max(last_frac, 1e-6), -0.24))

    if args.title:
        fig.suptitle(args.title, fontsize=15, y=0.982)

    png, pdf = out_stem + ".png", out_stem + ".pdf"
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s and %s", png, pdf)
    return png, pdf


def _legend_handles(rows: List[dict], args):
    # No skill swatches by default: the two tasks reuse the same eight hues for different skills, so
    # a shared swatch legend would be actively wrong. Skill identity is carried by text in every band
    # and in every film-strip title, so it is never colour-alone.
    handles = []
    if args.skill_legend:
        seen: List[Tuple[str, str]] = []
        for row in rows:
            for b in row["blocks"]:
                key = (b["label"], b["color"])
                if key not in seen and not b.get("is_replan"):
                    seen.append(key)
        handles += [Patch(fc=col, ec="none", label=lab) for lab, col in seen]
    if any(r["replan"] for r in rows):
        handles.append(Patch(fc=C_REPLAN, ec="none", label="Replanning (robot idle)"))
    handles.append(Line2D([0], [0], color=C_COMP, lw=1.9, label="completion prob"))
    if any(r["back"] is not None for r in rows):
        handles.append(Line2D([0], [0], color=C_BACK, lw=2.3, label="back prob"))
        handles.append(Line2D([0], [0], color=C_TAU_BACK, ls="--", lw=1.4, label="τ_back"))
    handles.append(Line2D([0], [0], color=C_TAU_DONE, ls="--", lw=1.4, label="τ_done"))
    handles.append(Line2D([0], [0], color=C_INK, ls=(0, (4, 2)), lw=1.0, label="pointer switch"))
    if any(r["recover"] for r in rows):
        handles.append(Line2D([0], [0], color=C_TAU_BACK, lw=2.4, label="recover"))
    handles.append(Patch(fc=C_WINDOW, alpha=0.16, ec="none", label="valid-switch window"))
    if any(r["back_gt"] for r in rows):
        handles.append(Patch(fc=C_BACK, alpha=0.22, ec="none", label="human interference"))
    handles.append(Patch(fc=C_WARM, alpha=0.16, ec="none", label="history warm-up"))
    return handles


# ──────────────────────────────── PPTX ──────────────────────────────────────────
def render_curve_strip(row, args, path, width_in, height_in):
    """Transparent, edge-to-edge curve strip for one row.

    The axes fill the whole canvas (``add_axes([0, 0, 1, 1])``) and the file is saved WITHOUT
    ``bbox_inches="tight"``: tight crops to the drawn artists, so the saved image's x-extent would no
    longer be the row's [lo, hi) span and the strip would drift out of alignment with the native
    blocks placed above it.
    """
    fig = plt.figure(figsize=(width_in, height_in), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    _draw_signals(ax, row, args, decorate=False)
    fig.savefig(path, dpi=200, transparent=True)
    plt.close(fig)
    return path


def save_thumbnail(img: np.ndarray, path: str) -> str:
    from PIL import Image
    Image.fromarray(img).save(path)
    return path


def render_pptx(rows: List[dict], args, out_path: str, assets_dir: str) -> str:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.util import Emu, Inches, Pt

    INK = RGBColor(0x1A, 0x1A, 0x1A)
    MUTE = RGBColor(0x6A, 0x6A, 0x66)
    LINE = RGBColor(0xDF, 0xDF, 0xDB)

    def _blank(prs):
        return prs.slides.add_slide(prs.slide_layouts[6])

    def _text(slide, x, y, w, h, text, *, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
              spacing=1.0):
        box = slide.shapes.add_textbox(x, y, w, h)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        lines = text.split("\n")
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = spacing
            run = p.add_run()
            run.text = line
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            run.font.name = "Calibri"
        return box

    def _header(slide, title, subtitle=""):
        _text(slide, Inches(0.62), Inches(0.40), Inches(12.1), Inches(0.6), title, size=27, bold=True)
        if subtitle:
            _text(slide, Inches(0.62), Inches(1.02), Inches(12.1), Inches(0.4), subtitle,
                  size=13, color=MUTE)
        ln = slide.shapes.add_shape(1, Inches(0.62), Inches(1.44), Inches(12.1), Emu(11430))
        ln.fill.solid(); ln.fill.fore_color.rgb = LINE; ln.line.fill.background()
        ln.shadow.inherit = False

    def _rgb(hex_str):
        h = hex_str.lstrip("#")
        return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    os.makedirs(assets_dir, exist_ok=True)
    span_max = max(r["hi"] - r["lo"] for r in rows)

    prs = Presentation()
    slide_w = Inches(args.pptx_width)
    slide_h = Inches(args.pptx_width * 9 / 16)
    prs.slide_width, prs.slide_height = slide_w, slide_h
    s = _blank(prs)

    scale = args.pptx_width / 13.333
    LEFT = int(Inches(0.62) * scale)
    FULL_W = slide_w - 2 * LEFT
    _header(s, args.pptx_title, args.pptx_subtitle)
    for shp in list(s.shapes):                       # rescale the 13.33in-authored header
        shp.left = int(shp.left * scale); shp.top = int(shp.top * scale)
        shp.width = int(shp.width * scale); shp.height = int(shp.height * scale)
        for para in getattr(shp, "text_frame", None).paragraphs if shp.has_text_frame else []:
            for run in para.runs:
                if run.font.size is not None:
                    run.font.size = Pt(run.font.size.pt * scale)

    row_h = int((slide_h - int(Inches(1.75) * scale) - int(Inches(0.55) * scale)) / len(rows))
    H_IMG = int(row_h * 0.48)
    H_BLK = int(row_h * 0.15)
    H_CRV = int(row_h * 0.30)
    GAP = int(row_h * 0.025)

    n_pics = 0
    for r, row in enumerate(rows):
        tm: TimeMap = row["tmap"]
        lo, hi, span_s = row["lo"], row["hi"], tm.span
        frac = ((hi - lo) / span_max) if args.shared_scale else 1.0
        row_w = int(FULL_W * frac)
        top = int(Inches(1.70) * scale) + r * row_h

        def x_of(frame, _w=row_w, _tm=tm):        # display-seconds aware, so elisions line up
            return LEFT + int(_w * _tm.t(frame) / max(_tm.span, 1e-9))

        _text(s, LEFT, top - int(Inches(0.30) * scale), int(FULL_W * 0.8),
              int(Inches(0.24) * scale),
              f"({'abcd'[r]})  {row['caption']}   ·   {span_s:.1f} s",
              size=12 * scale, bold=True, color=MUTE)

        # ── cam_top thumbnails: evenly spaced, so a short block never squashes its photo ──
        n = len(row["blocks"])
        gap_t = int(Inches(0.08) * scale)
        tw = int((FULL_W - gap_t * (n - 1)) / n)
        th = min(int(tw * 480 / 640), H_IMG)
        for c, b in enumerate(row["blocks"]):
            x = LEFT + c * (tw + gap_t)
            y = top + (H_IMG - th) // 2
            if b["key_frame"] is not None:
                p = save_thumbnail(b["key_frame"],
                                   os.path.join(assets_dir, f"{row['name']}_{b['start']:05d}.png"))
                pic = s.shapes.add_picture(p, x, y, width=tw, height=th)
                pic.line.color.rgb = _rgb(b["color"]); pic.line.width = Pt(2.0)
                n_pics += 1
            else:
                ph = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, tw, th)
                ph.fill.solid(); ph.fill.fore_color.rgb = _rgb(C_PLACEHOLDER)
                ph.line.color.rgb = _rgb(b["color"]); ph.line.width = Pt(2.0)
                ph.shadow.inherit = False

        # ── native, editable block strip ──
        blk_top = top + H_IMG + GAP
        for b in row["blocks"]:
            x0, x1 = x_of(b["start"]), x_of(b["end"])
            shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x0, blk_top,
                                     max(x1 - x0 - Emu(9144), Emu(72000)), H_BLK)
            shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(b["color"])
            shp.line.fill.background(); shp.shadow.inherit = False
            try:
                shp.adjustments[0] = 0.14
            except (IndexError, KeyError):
                pass
            tf = shp.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Emu(18288)
            tf.margin_top = tf.margin_bottom = 0
            p0 = tf.paragraphs[0]; p0.alignment = PP_ALIGN.CENTER
            run = p0.add_run(); run.text = b["label"]
            run.font.size = Pt(10.0 * scale); run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); run.font.name = "Calibri"
            p1 = tf.add_paragraph(); p1.alignment = PP_ALIGN.CENTER
            r1 = p1.add_run(); r1.text = f"f{b['start']}–{b['end']}"
            r1.font.size = Pt(7.0 * scale); r1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r1.font.name = "Calibri"

        # Break notation over the block that had a slice elided — without it the cut is invisible in
        # the deck, since a native rectangle carries no axis furniture of its own.
        for xb_s in tm.breaks:
            xb = LEFT + int(row_w * xb_s / max(tm.span, 1e-9))
            gap = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, xb - Emu(27432), blk_top - Emu(9144),
                                     Emu(54864), H_BLK + Emu(18288))
            gap.fill.solid(); gap.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            gap.line.fill.background(); gap.shadow.inherit = False
            for off in (-Emu(16000), Emu(16000)):
                sl = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, xb + off - Emu(4572),
                                        blk_top - Emu(9144), Emu(9144), H_BLK + Emu(18288))
                sl.fill.solid(); sl.fill.fore_color.rgb = INK
                sl.line.fill.background(); sl.shadow.inherit = False
                sl.rotation = 20

        # ── curve strip picture, x-aligned to the block strip ──
        crv_top = blk_top + H_BLK + GAP
        strip = render_curve_strip(row, args,
                                   os.path.join(assets_dir, f"curve_{row['name']}.png"),
                                   width_in=row_w / 914400, height_in=H_CRV / 914400)
        s.shapes.add_picture(strip, LEFT, crv_top, width=row_w, height=H_CRV)
        n_pics += 1

        ticks = [(0.0, "0"), (args.tau_done, f"τ_done {args.tau_done:g}"), (1.0, "1.0")]
        if row["back"] is not None:
            ticks.insert(2, (args.tau_back, f"τ_back {args.tau_back:g}"))
        for frac_y, lab in ticks:
            yy = crv_top + int(H_CRV * (1 - frac_y)) - int(Inches(0.09) * scale)
            col = (_rgb(C_TAU_DONE) if "done" in lab else
                   _rgb(C_TAU_BACK) if "back" in lab else MUTE)
            _text(s, LEFT - int(Inches(0.60) * scale), yy, int(Inches(0.56) * scale),
                  int(Inches(0.18) * scale), lab, size=7.5 * scale, color=col,
                  align=PP_ALIGN.RIGHT)

        # ── recover marker; the callout itself is opt-in, since the plot area is meant to be left
        #    clear for annotation by hand ──
        for f in row["recover"]:
            if not (lo <= f < hi):
                continue
            xr = x_of(f)
            bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, xr - Emu(11430), blk_top,
                                     Emu(22860), H_BLK + GAP + H_CRV)
            bar.fill.solid(); bar.fill.fore_color.rgb = _rgb(C_TAU_BACK)
            bar.line.fill.background(); bar.shadow.inherit = False
            if not args.pptx_callout:
                continue
            box_w, box_h = int(Inches(3.6) * scale), int(Inches(0.62) * scale)
            bx = min(xr + int(Inches(0.18) * scale), LEFT + FULL_W - box_w)
            by = crv_top + H_CRV - box_h - int(Inches(0.04) * scale)
            tb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, box_w, box_h)
            tb.fill.solid(); tb.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            tb.line.color.rgb = _rgb(C_TAU_BACK); tb.line.width = Pt(0.75)
            tb.shadow.inherit = False
            tf = tb.text_frame; tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Emu(45720)
            tf.margin_top = tf.margin_bottom = 0
            for i, line in enumerate(INTERVENTION_TEXT.split("\n")):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.LEFT
                run = p.add_run(); run.text = line
                run.font.size = Pt(8.0 * scale); run.font.color.rgb = INK
                run.font.name = "Calibri"
            conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, xr, by + box_h // 2,
                                          bx, by + box_h // 2)
            conn.line.color.rgb = _rgb(C_TAU_BACK); conn.line.width = Pt(1.0)

    # ── native legend ──
    # Same content as the PNG legend: no skill swatches unless asked, because the two rows reuse the
    # same hues for different skills. Entries wrap to a second line rather than running off the slide.
    entries: List[Tuple[str, str, str]] = []          # (label, colour, "patch" | "line")
    if args.skill_legend:
        for row in rows:
            for b in row["blocks"]:
                if not b.get("is_replan") and (b["label"], b["color"], "patch") not in entries:
                    entries.append((b["label"], b["color"], "patch"))
    if any(r["replan"] for r in rows):
        entries.append(("Replanning (robot idle)", C_REPLAN, "patch"))
    entries.append(("completion prob", C_COMP, "line"))
    if any(r["back"] is not None for r in rows):
        entries.append(("back prob", C_BACK, "line"))
    if any(r["recover"] for r in rows):
        entries.append(("recover", C_TAU_BACK, "line"))
    entries.append(("pointer switch", C_INK, "line"))
    entries.append(("valid-switch window", "#d3efe2", "patch"))
    if any(r["back_gt"] for r in rows):
        entries.append(("human interference", "#f9dde7", "patch"))
    entries.append(("history warm-up", "#e5e5e2", "patch"))

    leg_h = int(Inches(0.20) * scale)
    x, leg_y = LEFT, slide_h - int(Inches(0.46) * scale)
    for lab, col, kind in entries:
        need = int(Inches(0.25 + 0.20 * len(lab) * 0.42) * scale)
        if x + need > LEFT + FULL_W:                  # wrap instead of overflowing the slide
            x, leg_y = LEFT, leg_y + leg_h
        if kind == "patch":
            sw = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, leg_y + int(Inches(0.03) * scale),
                                    int(Inches(0.18) * scale), int(Inches(0.13) * scale))
        else:
            sw = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, leg_y + int(Inches(0.08) * scale),
                                    int(Inches(0.18) * scale), int(Inches(0.035) * scale))
        sw.fill.solid(); sw.fill.fore_color.rgb = _rgb(col)
        sw.line.fill.background(); sw.shadow.inherit = False
        _text(s, x + int(Inches(0.23) * scale), leg_y, int(Inches(1.5) * scale), leg_h,
              lab, size=8 * scale, color=MUTE)
        x += need

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    prs.save(out_path)
    logger.info("Wrote %s (%d pictures, %d native blocks)", out_path, n_pics,
                sum(len(r["blocks"]) for r in rows))
    return out_path


# ──────────────────────────────── CLI ───────────────────────────────────────────
def _parse_row(spec: str) -> Tuple[str, str, int, int]:
    parts = spec.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--row expects <name>:<pkl>:<seg_start>:<seg_count>, got {spec!r}")
    return parts[0], parts[1], int(parts[2]), int(parts[3])


def _parse_elide(spec: str) -> Tuple[str, int, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--elide expects <row>:<block_start_frame>:<seconds>, got {spec!r}")
    return parts[0], int(parts[1]), float(parts[2])


def _parse_rate(spec: str) -> Tuple[str, str]:
    parts = spec.split(":")
    if len(parts) != 2 or parts[1] not in ("auto", "tick", "per_frame"):
        raise argparse.ArgumentTypeError(
            f"--filter_rate expects <row>:<auto|tick|per_frame>, got {spec!r}")
    return parts[0], parts[1]


def _parse_stride(spec: str) -> Tuple[str, int]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--tick_stride_row expects <row>:<frames>, got {spec!r}")
    return parts[0], int(parts[1])


def _parse_fetch(spec: str) -> Tuple[str, str, int, str]:
    parts = spec.rsplit(":", 2)
    head = parts[0].split(":", 1)
    if len(parts) != 3 or len(head) != 2:
        raise argparse.ArgumentTypeError(
            f"--fetch expects <row>:<repo_id>:<episode>:<camera>, got {spec!r}")
    return head[0], head[1], int(parts[1]), parts[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", action="append", required=True, type=_parse_row,
                    help="<name>:<pkl>:<seg_start>:<seg_count>; repeat for more rows")
    ap.add_argument("--out", required=True, help="output stem (writes .png/.pdf/.pptx)")
    ap.add_argument("--tau_back", type=float, default=0.8)
    ap.add_argument("--tau_done", type=float, default=0.6)
    ap.add_argument("--curve_source", choices=("auto", "tick", "per_frame", "raw"), default="auto",
                    help="temporal resolution of the drawn curve. 'tick' = one step per System-2 "
                         "decision (coarse, but the tau crossing coincides with the switch); "
                         "'per_frame' = the pipeline's median+EMA advanced every frame (smooth and "
                         "fine, crosses tau before the tick that acts on it); 'raw' = unfiltered "
                         "per-frame sigmoid. Affects only what is plotted.")
    ap.add_argument("--filter_rate", action="append", type=_parse_rate, default=None,
                    help="<row>:<auto|tick|per_frame> — how often to advance the output filter when "
                         "reconstructing the curve. 'auto' (default) matches whatever the recorded "
                         "run did, which is the only setting whose curve explains its own switch "
                         "frames. Deployment is 'tick'.")
    ap.add_argument("--k_a", type=int, default=1,
                    help="advance debounce, used to rebuild the controller view of a raw cache")
    ap.add_argument("--cooldown", type=int, default=6,
                    help="ticks blocked after a switch, used the same way. 6 matches the bloodgas "
                         "run; at 10 the latch outlives two of the tubesort plateaus entirely and "
                         "their switches end up with no visible rise at all.")
    ap.add_argument("--tick_stride", type=int, default=15,
                    help="System-2 tick period in frames; sets how fast a consumed plateau is "
                         "released back into view (release takes about (cooldown + 2) x stride "
                         "frames, so a SMALLER stride shows MORE of the curve)")
    ap.add_argument("--tick_stride_row", action="append", type=_parse_stride, default=None,
                    help="<row>:<frames> — override --tick_stride for one row")
    ap.add_argument("--k_b", type=int, default=3, help="back debounce, for the back latch")
    ap.add_argument("--warmup_frames", type=int, default=60,
                    help="history warm-up length when the cache does not record one")
    ap.add_argument("--replan_end", default="auto",
                    help="'auto' = end of the interference window, or an absolute frame index")
    ap.add_argument("--replan_fallback_s", type=float, default=3.0,
                    help="replanning length when the recover has no interference window")
    ap.add_argument("--replan_frame", default="auto",
                    help="frame to photograph for the Replanning block; 'auto' = middle of the "
                         "interference window. The physical intervention often precedes the "
                         "annotated window, so an explicit frame usually reads better.")
    ap.add_argument("--replan_strip_label", default="Manual intervention",
                    help="film-strip caption for the Replanning block (the band keeps 'Replanning')")
    ap.add_argument("--post_recover_label", default="Grasp next blood sample",
                    help="label for the block that follows the Replanning block ('' to keep)")
    ap.add_argument("--no_replan", action="store_true", help="do not insert a Replanning block")
    ap.add_argument("--elide", action="append", type=_parse_elide, default=None,
                    help="<row>:<block_start_frame>:<seconds> — cut that many seconds out of the "
                         "middle of one subtask block so it stops dominating the timeline. The cut "
                         "is marked with break slashes; time stays linear either side of it.")
    ap.add_argument("--shared_scale", action="store_true",
                    help="draw every row at the same seconds-per-inch (shorter rows end short of "
                         "the right margin). Off by default: rows fill the width equally.")
    ap.add_argument("--fig_w", type=float, default=22.0)
    ap.add_argument("--fig_h", type=float, default=10.4)
    ap.add_argument("--lw_scale", type=float, default=1.0,
                    help="multiplier on every curve/threshold/event line width")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--legend_cols", type=int, default=10)
    ap.add_argument("--skill_legend", action="store_true",
                    help="also put a swatch per subtask in the legend (the two tasks reuse hues, "
                         "so this is off by default)")
    ap.add_argument("--fetch", action="append", type=_parse_fetch, default=None,
                    help="<row>:<repo_id>:<episode>:<camera> — decode the thumbnails the caches "
                         "do not hold (Replanning / post-recover). Memoised next to the outputs.")
    ap.add_argument("--title", default="")
    ap.add_argument("--pptx_width", type=float, default=13.333,
                    help="slide width in inches (16:9); widen for larger thumbnails")
    ap.add_argument("--pptx_title", default="System-2 closed-loop subtask control")
    ap.add_argument("--pptx_subtitle",
                    default="one row per task · every block below is native, editable "
                            "PowerPoint · curves and photos are pictures")
    ap.add_argument("--pptx_callout", action="store_true",
                    help="also place the manual-intervention textbox in the deck")
    ap.add_argument("--no_pptx", action="store_true")
    ap.add_argument("--no_png", action="store_true")
    args = ap.parse_args()

    args.tick_stride_by = dict(args.tick_stride_row or [])
    args.filter_rate_by = dict(args.filter_rate or [])
    rows = [load_row(name, pkl, s0, n, args) for name, pkl, s0, n in args.row]
    fetch_frames(rows, args.fetch, args.out + "_assets/extra_frames.pkl")
    apply_elisions(rows, args.elide)
    for row in rows:
        logger.info("[%s] %d blocks over f%d–%d, %.1f s shown: %s", row["name"],
                    len(row["blocks"]), row["lo"], row["hi"], row["tmap"].span,
                    " → ".join(b["label"] for b in row["blocks"]))

    if not args.no_png:
        render_png(rows, args, args.out)
    if not args.no_pptx:
        render_pptx(rows, args, args.out + ".pptx", args.out + "_assets")


if __name__ == "__main__":
    main()
