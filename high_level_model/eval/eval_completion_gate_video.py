"""Eval: visualise per-frame completion-gate predictions on a LeRobot episode.

Renders a side-by-side MP4:
  LEFT  : camera views (cam_top | cam_left | cam_right) with text overlays
  RIGHT : completion/back-prob chart — predicted completion probability (gamma_t),
          predicted back probability (beta_t), GT completion/back shading, segment
          dividers, moving cursor.

Usage:
    python -m high_level_model.eval.eval_completion_gate_video \\
        --repo_id /path/to/TubeSort_dagger_20260718 \\
        --gate_ckpt outputs/completion_gate/completion_gate_best.pth \\
        --episode 0 --gpu 0
"""

import argparse
import glob
import logging
import os
import yaml
from typing import List, Optional, Set, Tuple

import cv2
import imageio
import numpy as np
import torch
from torchvision.transforms import v2 as T
from tqdm import tqdm

from high_level_model.models.completion_gate import CompletionGate
from high_level_model.models.siglip_encoder import build_siglip_encoder
from high_level_model.data.completion_gate_dataset import CompletionGateDataset, Segment
from high_level_model.data.high_level_dataset import _get_ep_bounds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── layout constants ───────────────────────────────────────────────────────────
PANEL_W    = 620   # chart panel width
PANEL_H    = 480   # chart panel height (also target height for camera strip)
PAD_L      = 64    # chart area left margin (y-axis labels)
PAD_R      = 20
PAD_T      = 44
PAD_B      = 52

# BGR colours
C_BACK    = (  50,  50, 220)   # red     – back probability
C_CURSOR  = ( 200, 200,   0)   # cyan    – current-frame cursor
C_DIVIDER = ( 140, 140, 140)   # gray    – segment dividers
C_GRID    = (  55,  55,  55)   # dark    – grid
C_TEXT    = ( 225, 225, 225)   # near-white
C_BG      = (  22,  22,  22)   # near-black background
C_GTBACK  = (  60,  40,  90)   # dark purple – GT back annotation shading
C_COMP    = ( 230,  90, 200)   # magenta – predicted completion probability
C_GTCOMP  = (  40,  70,  40)   # dark green – GT completion window shading
C_TAU     = ( 120, 220, 120)   # green   – tau_done threshold line
C_SWITCH  = (  60, 200, 255)   # amber   – frame where the POINTER advanced
C_RECOVER = (  80, 120, 255)   # orange  – frame where the pointer recovered

# How long (frames) the ">> ADVANCE" banner stays on screen after a pointer switch.
BANNER_FRAMES = 25


# ── helpers ────────────────────────────────────────────────────────────────────

def _chart_xy(local_f: int, win_start: int, win_len: int, value: float) -> Tuple[int, int]:
    """Map (frame_index, value∈[0,1]) → pixel (x, y) inside the chart area.

    The x axis spans the sliding window ``[win_start, win_start + win_len)`` rather than the whole
    episode: at 3600 frames across ~540 px an entire subtask collapses into a few pixels and the
    completion curve's shape — the thing the controller actually thresholds — is invisible.
    """
    chart_w = PANEL_W - PAD_L - PAD_R
    chart_h = PANEL_H - PAD_T - PAD_B
    x = PAD_L + int((local_f - win_start) / max(1, win_len - 1) * chart_w)
    y = PAD_T + int((1.0 - float(value)) * chart_h)
    return x, y


def window_start(cur_local: int, win_len: int, follow_frac: float = 0.8) -> int:
    """Left edge of the sliding window for playhead ``cur_local``.

    The playhead runs freely to ``follow_frac`` of the window, then the window scrolls with it so the
    cursor stays pinned at that fraction and ``(1 - follow_frac)`` of the width keeps showing the
    curve ahead. Never negative, so the first frames of an episode still render against a full axis.
    """
    lead = int(win_len * follow_frac)
    return max(0, cur_local - lead)


def _put(img: np.ndarray, text: str, org, scale: float = 0.38,
         color=C_TEXT, thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def make_history_indices(f: int, ep_start: int, history_span: int,
                         history_skip_frame: int, history_len: int) -> List[int]:
    """Mirror CompletionGateDataset's sampling; clamp to ep_start (no cross-episode)."""
    raw = list(range(f - history_span, f + 1, history_skip_frame))
    selected = raw[-history_len:]
    return [max(ep_start, idx) for idx in selected]


def frame_to_bgr(raw_ds, frame_idx: int, camera_names: List[str],
                 target_h: int) -> np.ndarray:
    """Return BGR uint8 image: cameras concatenated horizontally, all at target_h rows."""
    data = raw_ds[frame_idx]
    imgs = []
    for cam in camera_names:
        t = data[cam]                                          # [C, H, W] float32 [0,1]
        img = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        new_w = max(1, int(w * target_h / h))
        imgs.append(cv2.resize(img, (new_w, target_h)))
    return cv2.hconcat(imgs)


def add_overlay(canvas: np.ndarray, episode: int, local_f: int, total: int,
                skill: str, back_p: float, comp_p: float,
                show_back: bool = True, plan_pos: Optional[Tuple[int, int]] = None,
                action: Optional[str] = None, banner: Optional[Tuple[str, str]] = None) -> None:
    """Draw info lines onto the top-left of canvas (in-place).

    In closed loop ``plan_pos`` (1-based pointer, plan length), ``action`` (the controller's last
    decision) and ``banner`` (``(action, next_skill)`` for the few frames after a switch) turn the
    overlay from "what does the gate think" into "what did System 2 decide" — the pointer position
    is the *active atomic skill*, so a viewer can see it advance and see which signal drove it.
    """
    skill_disp = skill if len(skill) <= 52 else skill[:49] + "..."
    probs = (f"back_prob = {back_p:.3f}    completion_prob = {comp_p:.3f}"
             if show_back else f"completion_prob = {comp_p:.3f}")
    lines = [f"EP {episode}   frame {local_f:04d} / {total - 1:04d}"]
    if plan_pos is not None:
        lines[0] += f"   plan {plan_pos[0]}/{plan_pos[1]}"
        if action:
            lines[0] += f"   action: {action.upper()}"
    lines += [f"skill: {skill_disp}", probs]
    y = 24
    for line in lines:
        # shadow
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (230, 230, 230), 1, cv2.LINE_AA)
        y += 24

    if banner is not None:
        act, nxt = banner
        colour = C_RECOVER if act == "recover" else C_SWITCH
        arrow = "<< RECOVER" if act == "recover" else ">> ADVANCE"
        nxt_disp = nxt if len(nxt) <= 40 else nxt[:37] + "..."
        text = f"{arrow} -> {nxt_disp}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        by = y + 10
        cv2.rectangle(canvas, (6, by - th - 8), (14 + tw, by + 8), (0, 0, 0), -1)
        cv2.rectangle(canvas, (6, by - th - 8), (14 + tw, by + 8), colour, 2)
        cv2.putText(canvas, text, (10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colour, 2, cv2.LINE_AA)


# ── chart drawing ──────────────────────────────────────────────────────────────

def draw_chart(
    back_probs: List[float],
    comp_probs: List[float],
    seg_change_locals: List[int],
    seg_skills:        List[str],
    back_gt_locals:    Set[int],
    comp_gt_locals:    Set[int],
    cur_local: int,
    total:     int,
    show_back: bool = True,
    tau_done: Optional[float] = None,
    switch_locals: Optional[List[Tuple[int, str]]] = None,
    win_len: int = 150,
    show_gt: bool = True,
) -> np.ndarray:
    """Return BGR uint8 chart panel of size (PANEL_H, PANEL_W).

    The x axis is a **sliding window** of ``win_len`` frames that follows the playhead (see
    :func:`window_start`), not the whole episode — over a few thousand frames the completion curve
    otherwise compresses to a few pixels per subtask and its shape near tau_done, which is the whole
    point of the panel, cannot be read.

    ``tau_done`` draws the advance threshold and ``switch_locals`` (``(frame, action)`` pairs) draws
    where the pointer actually switched. With ``show_gt`` the grey dividers mark GT segment
    boundaries, so the gap between an amber switch line and the grey line next to it *is* the timing
    error. Set ``show_gt=False`` for captures with no per-frame subtask labels, where those dividers
    would be fiction.
    """
    panel = np.full((PANEL_H, PANEL_W, 3), C_BG, dtype=np.uint8)
    chart_w = PANEL_W - PAD_L - PAD_R
    chart_h = PANEL_H - PAD_T - PAD_B
    ws = window_start(cur_local, win_len)
    we = ws + win_len

    def _visible_runs(locs: Set[int]):
        """Contiguous [s, e] runs of ``locs`` clipped to the current window."""
        sel = sorted(f for f in locs if ws <= f < we)
        i = 0
        while i < len(sel):
            s = e = sel[i]
            while i + 1 < len(sel) and sel[i + 1] == sel[i] + 1:
                i += 1
                e = sel[i]
            yield s, e
            i += 1

    # --- GT back annotation shading (bottom stripe) ---
    if show_gt and back_gt_locals:
        for s, e in _visible_runs(back_gt_locals):
            x0, _ = _chart_xy(s, ws, win_len, 0.0)
            x1, _ = _chart_xy(e, ws, win_len, 0.0)
            oy = PAD_T + chart_h - 16
            overlay = panel.copy()
            cv2.rectangle(overlay, (x0, oy), (x1, PAD_T + chart_h), C_GTBACK, -1)
            cv2.addWeighted(overlay, 0.5, panel, 0.5, 0, panel)

    # --- GT completion window shading (top stripe) ---
    if show_gt and comp_gt_locals:
        for s, e in _visible_runs(comp_gt_locals):
            x0, _ = _chart_xy(s, ws, win_len, 0.0)
            x1, _ = _chart_xy(e, ws, win_len, 0.0)
            overlay = panel.copy()
            cv2.rectangle(overlay, (x0, PAD_T), (x1, PAD_T + 16), C_GTCOMP, -1)
            cv2.addWeighted(overlay, 0.5, panel, 0.5, 0, panel)

    # --- horizontal grid + y-axis labels ---
    for yv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, py = _chart_xy(0, ws, win_len, yv)
        cv2.line(panel, (PAD_L, py), (PAD_L + chart_w, py), C_GRID, 1)
        _put(panel, f"{yv:.2f}", (4, py + 5))

    # --- x-axis frame ticks (the window scrolls, so the axis has to say where it is) ---
    step = max(10, (win_len // 5 // 10) * 10)
    for f in range(((ws + step - 1) // step) * step, we, step):
        x, _ = _chart_xy(f, ws, win_len, 0.0)
        cv2.line(panel, (x, PAD_T + chart_h), (x, PAD_T + chart_h + 4), C_TEXT, 1)
        _put(panel, str(f), (x - 12, PAD_T + chart_h + 16))

    # --- GT segment dividers (no skill text: it overlaps itself at this width and the active
    #     skill is already spelled out on the camera panel) ---
    if show_gt:
        for lf in seg_change_locals:
            if not (ws <= lf < we):
                continue
            x, _ = _chart_xy(lf, ws, win_len, 0.0)
            cv2.line(panel, (x, PAD_T), (x, PAD_T + chart_h), C_DIVIDER, 1)

    # --- chart border ---
    cv2.rectangle(panel,
                  (PAD_L, PAD_T),
                  (PAD_L + chart_w, PAD_T + chart_h),
                  C_TEXT, 1)

    # --- tau_done threshold (dashed green) ---
    if tau_done is not None:
        _, ty = _chart_xy(0, ws, win_len, tau_done)
        for x in range(PAD_L, PAD_L + chart_w, 12):
            cv2.line(panel, (x, ty), (min(x + 6, PAD_L + chart_w), ty), C_TAU, 1)
        _put(panel, f"tau_done={tau_done:.2f}", (PAD_L + 4, ty - 4), color=C_TAU)

    # --- pointer switches (where the controller actually advanced/recovered) ---
    for lf, act in (switch_locals or []):
        if lf > cur_local:
            break                      # don't reveal switches the playhead hasn't reached yet
        if not (ws <= lf < we):
            continue
        x, _ = _chart_xy(lf, ws, win_len, 0.0)
        cv2.line(panel, (x, PAD_T), (x, PAD_T + chart_h),
                 C_RECOVER if act == "recover" else C_SWITCH, 2)

    def _polyline(series: List[float], colour, thick: int) -> None:
        """Draw the part of ``series`` inside the window, one segment per frame step."""
        lo = max(1, ws)
        hi = min(len(series), we)
        pts = [_chart_xy(i, ws, win_len, series[i]) for i in range(lo - 1, hi)]
        if len(pts) > 1:
            cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, colour, thick,
                          cv2.LINE_AA)

    if show_back:
        _polyline(back_probs, C_BACK, 1)
    # predicted completion probability — the signal the advance rule thresholds, drawn last so it
    # stays on top of the back curve where the two overlap.
    _polyline(comp_probs, C_COMP, 2)

    # --- current value readout at the playhead ---
    if comp_probs:
        cval = comp_probs[min(cur_local, len(comp_probs) - 1)]
        cx, cy = _chart_xy(cur_local, ws, win_len, cval)
        cv2.circle(panel, (cx, cy), 4, C_COMP, -1)
        _put(panel, f"{cval:.2f}", (cx + 7, cy - 6), scale=0.45, color=C_COMP)

    # --- current-frame cursor ---
    cx, _ = _chart_xy(cur_local, ws, win_len, 0.0)
    cv2.line(panel, (cx, PAD_T), (cx, PAD_T + chart_h), C_CURSOR, 2)

    # --- legend (top-right) ---
    lx = PAD_L + chart_w - 175
    ly = PAD_T + 14
    legend = [(C_COMP, "completion", 2)]
    if show_back:
        legend.append((C_BACK, "back prob", 1))
    if switch_locals is not None:
        legend.append((C_SWITCH, "pointer switch", 2))
    if show_gt:
        legend.append((C_DIVIDER, "GT boundary", 1))
    for clr, label, thick in legend:
        cv2.line(panel, (lx, ly), (lx + 20, ly), clr, thick)
        _put(panel, label, (lx + 24, ly + 4))
        ly += 16

    # --- axis label ---
    _put(panel, f"frame  (window {win_len})", (PAD_L + chart_w // 2 - 48, PANEL_H - 8))

    return panel


def draw_replan_card(canvas: np.ndarray, elapsed: float, total_s: float,
                     plan_texts: List[str], pointer: int, replanned: bool,
                     style: str = "minimal") -> np.ndarray:
    """Dim ``canvas`` and overlay a 'System 2 replanning' card (returned as a new image).

    The real system pauses here: a recover hands the violated skill's pre/postcondition facts to the
    LLM and waits for a fresh plan. Rendering those seconds as a held frame keeps the video honest
    about that latency instead of cutting straight to the next skill as if replanning were free.

    ``style="minimal"`` (default) is exactly that and nothing more — a dimmed frame with one line of
    text, so the pause reads as *waiting* rather than as a UI animation competing with the robot
    footage. ``style="detailed"`` keeps the older card (progress bar + the plan that came back),
    which is useful when debugging what the replanner actually returned.
    """
    if style == "minimal":
        out = (canvas.astype(np.float32) * 0.30).astype(np.uint8)
        h, w = out.shape[:2]
        text = "SYSTEM 2 - REPLANNING" if replanned else "SYSTEM 2 - REPLAN BUDGET EXHAUSTED"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(out, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    C_RECOVER, 2, cv2.LINE_AA)
        return out

    out = (canvas.astype(np.float32) * 0.30).astype(np.uint8)
    h, w = out.shape[:2]
    cx = w // 2

    # Opaque backdrop: the card sits on top of whatever the camera/chart panels were drawing, and a
    # dim-only overlay leaves both texts legible-but-interleaved.
    card_w = min(w - 40, 760)
    card_h = min(h - 40, 150 + 19 * max(1, len(plan_texts)))
    x0, y0 = cx - card_w // 2, max(20, (h - card_h) // 2)
    x1, y1 = x0 + card_w, y0 + card_h
    cv2.rectangle(out, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), C_RECOVER, 2)

    title = "SYSTEM 2 - REPLANNING" if replanned else "SYSTEM 2 - RECOVER (replan budget exhausted)"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(out, title, (cx - tw // 2, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                C_RECOVER, 2, cv2.LINE_AA)

    sub = "precondition violated - re-routing remaining plan"
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(out, sub, (cx - sw // 2, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    # progress bar for the wait
    bx, bw, bh, by = x0 + 28, card_w - 56 - 90, 10, y0 + 82
    frac = 0.0 if total_s <= 0 else min(1.0, elapsed / total_s)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1)
    cv2.rectangle(out, (bx, by), (bx + int(bw * frac), by + bh), C_RECOVER, -1)
    _put(out, f"{elapsed:4.1f}s / {total_s:.0f}s", (bx + bw + 10, by + bh), scale=0.45)

    # the plan that came back, with the new pointer marked
    y = by + 34
    _put(out, "updated plan:", (bx, y), scale=0.5)
    y += 22
    for i, txt in enumerate(plan_texts):
        if y > y1 - 8:
            break
        active = (i == pointer)
        mark = ">>" if active else "  "
        colour = C_SWITCH if active else (170, 170, 170)
        disp = txt if len(txt) <= 58 else txt[:55] + "..."
        _put(out, f"{mark} {i + 1}. {disp}", (bx, y), scale=0.45, color=colour,
             thick=2 if active else 1)
        y += 19
    return out


# ── planner wiring (closed loop) ───────────────────────────────────────────────

def _build_planner(args, library):
    """The planner the pipeline replans with: Qwen when ``--use_llm``, offline otherwise.

    Defaults to offline on purpose. A video is a figure: the deterministic fallback
    (``cycle * remaining + terminal``) reruns to the same frames, whereas an LLM round-trip does
    not, and the thing being demonstrated here is the *controller*, not the planner's wording.
    """
    from high_level_model.planning.planner import Planner
    from high_level_model.planning.prompt_templates import load_prompt_set

    prompt_set = load_prompt_set(args.planner_prompt)
    if not args.use_llm:
        return Planner(library, prompt_set=prompt_set)
    from high_level_model.planning.llm_backends import qwen_llm_fn

    logger.info("Planner: Qwen backend (DASHSCOPE) with prompt set %s",
                args.planner_prompt or "<built-in default>")
    return Planner(library, llm_fn=qwen_llm_fn(), prompt_set=prompt_set)


def _make_remaining_cycles_fn(args, library, planner, frame_to_seg, ep_start, ep_end, cur_local):
    """Callable returning how many protocol cycles are still ahead of ``cur_local[0]``.

    With per-frame subtask labels (``--remaining_cycles auto``) this counts how many times the
    cycle's first skill still *starts* a GT segment after the current frame — i.e. how many
    unprocessed work items the episode is about to show. That is the number a scene-reading planner
    would have to produce on the robot, read here off the labels instead of off a VLM so the video
    is reproducible. An explicit integer pins it (and is required with ``--plan_from_library``,
    where there are no labels to count).
    """
    if args.remaining_cycles != "auto":
        n = max(1, int(args.remaining_cycles))
        logger.info("Remaining-cycles hook pinned to %d.", n)
        return lambda: n
    if args.plan_from_library:
        logger.warning("--remaining_cycles auto needs GT subtask labels, which --plan_from_library "
                       "disables; assuming 1 remaining cycle.")
        return lambda: 1

    cycle = planner.cycle_ids()
    first_text = library.instruction_of(cycle[0]) if cycle else None

    starts: List[Tuple[int, str]] = []          # (local frame, skill text) per GT segment start
    prev = None
    for f in range(ep_start, ep_end):
        seg = frame_to_seg.get(f)
        key = None if seg is None else seg.start
        if key != prev:
            starts.append((f - ep_start, "" if seg is None else seg.skill_text))
        prev = key

    def _count() -> int:
        n = sum(1 for lf, txt in starts if lf > cur_local[0] and txt == first_text)
        logger.info("Remaining-cycles hook: %d more '%s' segment(s) after local frame %d.",
                    n, first_text, cur_local[0])
        return max(1, n)

    return _count


# ── checkpoint loading ─────────────────────────────────────────────────────────

GATE_SIDECAR_NAME = "completion_gate_config.yaml"


def _resolve_gate_sidecar(ckpt_path: str) -> Tuple[str, dict]:
    """Resolve ``ckpt_path`` (run folder or ``.pth``) to ``(pth_path, sidecar_dict)``. ``sidecar_dict``
    is the FULL set of gate-reconstruction params written by training (``build_gate_sidecar``):
    ``camera_names``, ``history_len``, ``history_skip_frame`` and a nested ``model`` arch dict. Falls
    back to the config embedded in the ``.pth`` (where the data params live under ``dataset``), else
    ``{}`` (old checkpoint → caller defaults). Kept local so ``high_level_model`` never imports
    ``low_level_model``."""
    if os.path.isdir(ckpt_path):
        pth = os.path.join(ckpt_path, "completion_gate_best.pth")
        if not os.path.isfile(pth):
            cands = sorted(glob.glob(os.path.join(ckpt_path, "*.pth")), key=os.path.getmtime)
            if not cands:
                raise FileNotFoundError(f"No .pth checkpoint found in folder {ckpt_path!r}")
            pth = cands[-1]
    else:
        pth = ckpt_path
    sidecar = os.path.join(os.path.dirname(pth), GATE_SIDECAR_NAME)
    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            return pth, (yaml.safe_load(f) or {})
    # Fallback: the .pth embeds the full training config (dataclass dump) — flatten the data params
    # we care about out of its ``dataset``/``model`` groups into the same shape as the sidecar.
    try:
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        cfg = ck.get("config") if isinstance(ck, dict) else None
    except Exception:
        cfg = None
    if isinstance(cfg, dict):
        ds = cfg.get("dataset", {}) or {}
        return pth, {
            "camera_names": ds.get("camera_names"),
            "history_len": ds.get("history_len"),
            "prediction_offset": ds.get("prediction_offset"),
            "history_skip_frame": ds.get("history_skip_frame"),
            "completion_window": ds.get("completion_window"),
            "completion_at_episode_end": ds.get("completion_at_episode_end"),
            "model": cfg.get("model", {}) or {},
        }
    return pth, {}


def load_gate(ckpt_path: str, backbone, device) -> CompletionGate:
    ckpt_path, side = _resolve_gate_sidecar(ckpt_path)
    mc = side.get("model", {}) or {}
    gate = CompletionGate(
        backbone, freeze_siglip=True,
        temporal_layers=mc.get("temporal_layers", 2),
        temporal_heads=mc.get("temporal_heads", 8),
        use_cross_attention=mc.get("use_cross_attention", False),
        cross_attn_heads=mc.get("cross_attn_heads", 8),
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd_ck    = ck["state_dict"]
    sd_model = gate.state_dict()

    filtered, skipped = {}, []
    for k, v in sd_ck.items():
        if k in sd_model and sd_model[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(f"{k}{tuple(v.shape)}")

    if skipped:
        logger.info("Skipped on load (shape mismatch / not in gate): %s", skipped)

    missing, _ = gate.load_state_dict(filtered, strict=False)
    if missing:
        logger.info("Keys kept at init (not in checkpoint): %s", missing[:10])

    gate.eval()
    # Train-only runs (no val split) store the metric under ``train_*``; prefer whichever the
    # checkpoint actually carries so the log isn't a bare nan.
    split = "val" if ck.get("val_completion_bce") is not None else "train"
    logger.info("Gate loaded  epoch=%s  %s_completion_bce=%.5f",
                ck.get("epoch", "?"), split, ck.get(f"{split}_completion_bce", float("nan")))
    return gate


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise CompletionGate predictions on a LeRobot episode."
    )
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--gate_ckpt", type=str, required=True)
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--camera_names", nargs="+", default=None)
    parser.add_argument("--history_len",       type=int,   default=None)
    parser.add_argument("--prediction_offset", type=int,   default=None)
    parser.add_argument("--history_skip_frame",type=int,   default=None)
    parser.add_argument("--closed_loop", action="store_true", default=False,
                        help="Drive the skill text from a System2Pipeline pointer instead of the GT "
                             "segment (oracle). Reflects deployment: once the pointer is wrong, the "
                             "skill text stays wrong. Requires --skill_library.")
    parser.add_argument("--skill_library", type=str, default=None,
                        help="Skill-library YAML (configs/skill_library/*.yaml); needed for --closed_loop")
    # Controller thresholds for --closed_loop. NOTE these tick once per FRAME here, whereas on the
    # robot System 2 ticks every `sampling_interval` frames — so k_a/cooldown are NOT directly
    # comparable to inference.yaml unless you also set --tick_stride to that sampling_interval.
    parser.add_argument("--tau_done", type=float, default=0.6)
    parser.add_argument("--k_a", type=int, default=2)
    parser.add_argument("--cooldown", type=int, default=5)
    # Recovery is OFF by default (build_eval_pipeline pins tau_back above 1 and replan_budget to 0)
    # so a mid-episode replan can't rewrite the oracle plan while advance timing is being measured.
    # Pass these to switch it back on for a task whose back head is actually supervised.
    parser.add_argument("--tau_back", type=float, default=None,
                        help="Enable recovery with this back-probability threshold (default: "
                             "recovery disabled). Needs a back head that was actually trained.")
    parser.add_argument("--k_b", type=int, default=3,
                        help="k-of-k debounce for recover (only used with --tau_back)")
    parser.add_argument("--replan_budget", type=int, default=0,
                        help="Max LLM replans per episode after a recover (only with --tau_back)")
    parser.add_argument("--plan_from_library", action="store_true", default=False,
                        help="Take the plan from the skill library's declaration order instead of "
                             "deriving an oracle plan from the GT segments. Required for captures "
                             "with no per-frame subtask labels, and closer to deployment (that is "
                             "the planner's own fallback when no LLM key is set). Disables the GT "
                             "overlays and the switch-timing report, which need labels to mean "
                             "anything.")
    parser.add_argument("--chart_window", type=int, default=150,
                        help="Sliding-window width (frames) of the chart x-axis. The playhead runs "
                             "to 80%% of the window, then the window scrolls with it.")
    parser.add_argument("--replan_hold_s", type=float, default=0.0,
                        help="On each recover, hold the video for this many seconds on a "
                             "'REPLANNING' card to represent the LLM replan latency the real system "
                             "pays. 0 disables (the video then matches wall-clock episode time).")
    parser.add_argument("--replan_card", choices=["minimal", "detailed"], default="minimal",
                        help="Look of the replan hold. 'minimal' = dimmed frame + one line, so the "
                             "pause just reads as waiting. 'detailed' adds a progress bar and the "
                             "plan the replanner returned (debugging).")
    parser.add_argument("--remaining_cycles", type=str, default="auto",
                        help="How many work items (samples / tubes) are still to be processed when "
                             "a recover fires — the replanner repeats the protocol cycle that many "
                             "times. 'auto' counts, in the GT segments after the recover frame, how "
                             "often the cycle's first skill still starts a segment. Give an integer "
                             "to pin it (required with --plan_from_library, which has no labels).")
    parser.add_argument("--use_llm", action="store_true", default=False,
                        help="Route plan/replan through the Qwen backend (needs DASHSCOPE_API_KEY). "
                             "Off by default: the deterministic fallback is reproducible, which is "
                             "what a figure/video wants.")
    parser.add_argument("--planner_prompt", type=str, default=None,
                        help="Planner prompt-set YAML (configs/skill_library/prompts/*.yaml); only "
                             "meaningful together with --use_llm.")
    parser.add_argument("--no_back_curve", action="store_true", default=False,
                        help="Hide beta_t from the chart and overlay. Use it for tasks whose "
                             "back_event column is all-zero (the head never learned, so its output "
                             "is noise and showing it implies a signal that isn't there).")
    parser.add_argument("--tick_stride", type=int, default=15,
                        help="run the pointer once every N frames, matching the deployment "
                             "sampling_interval so k_a/cooldown mean the same as in inference.yaml")
    parser.add_argument("--output", type=str, default=None,
                        help="Output mp4 path (default: "
                             "<ckpt_dir>/videos/<run>_<ckpt>_ep{N:04d}.mp4)")
    parser.add_argument("--fps",        type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cam_h",      type=int, default=480,
                        help="Display height (px) for each camera image")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tolerance_s", type=float, default=None,
                        help="lerobot video-frame timestamp tolerance (s). Raise above the default "
                             "1e-4 for captures with timestamp drift (e.g. 1e-2 for TubeSort_20260720).")
    parser.add_argument("--back_annotations", type=str, default=None,
                        help="Optional JSON: [{episode, frame_start, frame_end}, ...] for GT back shading")
    parser.add_argument("--completion_window", type=int, default=None,
                        help="GT completion = last N frames of each completed segment "
                             "(default: read from the checkpoint sidecar to match training)")
    parser.add_argument("--completion_at_episode_end", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Also count the episode's final segment as completed when drawing the "
                             "GT completion window (default: from the sidecar, to match training)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    pth_path, side = _resolve_gate_sidecar(args.gate_ckpt)
    _DEF = {"camera_names": ["observation.images.cam_top",
                             "observation.images.cam_left",
                             "observation.images.cam_right"],
            "history_len": 6, "prediction_offset": 0, "history_skip_frame": 10,
            "completion_window": 20, "completion_at_episode_end": False}
    for k in _DEF:
        if getattr(args, k) is None:
            setattr(args, k, side.get(k) if side.get(k) is not None else _DEF[k])
    logger.info("Eval data params (sidecar/CLI): camera_names=%s history_len=%d "
                "prediction_offset=%d history_skip_frame=%d "
                "completion_window=%d completion_at_episode_end=%s",
                args.camera_names, args.history_len, args.prediction_offset,
                args.history_skip_frame, args.completion_window, args.completion_at_episode_end)
    if args.closed_loop and not args.skill_library:
        parser.error("--closed_loop requires --skill_library")

    if args.output:
        out_path = args.output
    else:
        # Anchor videos/ to the *run folder* (the dir holding the resolved .pth + sidecar), so the
        # output lands at .../{run_name}/videos/... whether --gate_ckpt points at the run folder or
        # directly at the .pth. (dirname(gate_ckpt) would point one level too high for a folder arg.)
        run_dir   = os.path.dirname(os.path.abspath(pth_path))
        run_name  = os.path.basename(run_dir)                              # the training run folder
        ckpt_stem = os.path.splitext(os.path.basename(pth_path))[0]        # e.g. completion_gate_best
        out_path = os.path.join(
            run_dir, "videos", f"{run_name}_{ckpt_stem}_ep{args.episode:04d}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── 1. datasets ────────────────────────────────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _tol = {} if args.tolerance_s is None else {"tolerance_s": args.tolerance_s}
    model_ds = LeRobotDataset(
        repo_id=args.repo_id,
        image_transforms=T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        **_tol,
    )
    raw_ds = LeRobotDataset(
        repo_id=args.repo_id,
        image_transforms=T.Compose([T.ToDtype(torch.float32, scale=True)]),
        **_tol,
    )

    # ── 2. segment / GT metadata ───────────────────────────────────────────────
    gate_ds = CompletionGateDataset(
        model_ds,
        subset_episodes=[args.episode],
        camera_names=args.camera_names,
        history_len=args.history_len,
        prediction_offset=args.prediction_offset,
        history_skip_frame=args.history_skip_frame,
        use_command_in_meta=False,
        back_annotations_path=args.back_annotations,
        completion_window=args.completion_window,
        completion_at_episode_end=args.completion_at_episode_end,
    )
    task_instruction = gate_ds.get_task_instruction(args.episode)
    frame_to_seg    = gate_ds.frame_to_seg        # abs_frame → Segment(start, end, skill_text, ...)
    back_gt_abs     = gate_ds.back_frames         # set of abs frame indices
    comp_gt_abs     = gate_ds.completion_frames   # set of abs frame indices (GT completion windows)

    ep_start, ep_end = _get_ep_bounds(model_ds.meta, args.episode)
    total_frames = ep_end - ep_start
    logger.info("Episode %d: [%d, %d)  total=%d frames", args.episode, ep_start, ep_end, total_frames)
    logger.info("Task instruction: %s", task_instruction)

    # ── 3. build + load model ──────────────────────────────────────────────────
    backbone = build_siglip_encoder(device, args.history_len, freeze_siglip=True).to(device)
    gate = load_gate(args.gate_ckpt, backbone, device)

    history_span = args.history_len * args.history_skip_frame

    # ── 4. inference (full episode) ───────────────────────────────────────────
    pipeline = None
    if args.closed_loop:
        # Closed loop: the pointer picks the skill text, so a wrong advance keeps feeding the wrong
        # skill for the rest of the episode — exactly what happens on the robot. The pointer is
        # stateful, so this path must run strictly frame-by-frame (no batching).
        from high_level_model.planning.skill_library import load_skill_library
        from high_level_model.eval.gate_metrics import (build_eval_pipeline,
                                                        oracle_plan_from_segments)

        library = load_skill_library(args.skill_library)
        if args.plan_from_library:
            # No usable per-frame labels (or a deliberately deployment-shaped run): the plan is the
            # library's declaration order, exactly what Planner falls back to without an LLM key.
            plan_ids = list(library.ids())
            plan_texts = [library.instruction_of(sid) for sid in plan_ids]
            logger.info("Plan taken from the skill library declaration order (%d skills); GT "
                        "overlays and the switch-timing report are disabled.", len(plan_ids))
        else:
            # Oracle plan + recovery disabled — see gate_metrics.build_eval_pipeline. Without both,
            # the planner's declaration-order fallback cannot express a repeated grasp/place cycle,
            # and a mid-episode replan would rewrite the plan out from under the measurement.
            plan_ids, plan_texts = oracle_plan_from_segments(frame_to_seg, ep_start, ep_end, library)
        recover_kw = ({} if args.tau_back is None else
                      {"tau_back": args.tau_back, "k_b": args.k_b,
                       "replan_budget": args.replan_budget})
        pipeline = build_eval_pipeline(
            library, plan_ids, task_instruction, planner=_build_planner(args, library),
            tau_done=args.tau_done, k_a=args.k_a, cooldown=args.cooldown, **recover_kw)
        pipeline.gate = gate
        # How much work is left when a recover fires. Without this the replanner assumes a single
        # remaining cycle, and an episode that still has several samples on the bench gets a plan
        # that runs out long before the footage does.
        cur_local = [0]
        remaining_fn = _make_remaining_cycles_fn(
            args, library, pipeline.planner, frame_to_seg, ep_start, ep_end, cur_local)
        pipeline.remaining_cycles_fn = remaining_fn
        logger.info("Closed-loop eval: %d-step oracle plan, tau_done=%.2f k_a=%d cooldown=%d, "
                    "pointer ticked every %d frames (= deployment sampling_interval, so k_a and "
                    "cooldown mean the same here as in inference.yaml)",
                    len(plan_ids), args.tau_done, args.k_a, args.cooldown, args.tick_stride)
        logger.info("Recovery %s", "disabled (advance timing only)" if args.tau_back is None else
                    f"ENABLED: tau_back={args.tau_back:.2f} k_b={args.k_b} "
                    f"replan_budget={args.replan_budget}")

    logger.info("%s inference over %d frames (batch_size=%d) …",
                "Closed-loop" if args.closed_loop else "Batch", total_frames,
                1 if args.closed_loop else args.batch_size)

    back_probs_all:  List[float] = []
    comp_probs_all:  List[float] = []
    skill_texts_all: List[str]   = []

    seg_change_locals: List[int] = []    # local frame index where a new segment starts
    seg_skills:        List[str] = []    # skill text at each segment change

    # closed-loop only: where the *pointer* switched skills, for comparison against the GT boundaries
    ptr_change_locals: List[int] = []
    ptr_skills:        List[str] = []
    # per-frame controller decision + the (frame, action) pairs the chart marks and the banner fires on
    actions_all:   List[str] = []
    plan_pos_all:  List[Tuple[int, int]] = []
    switch_locals: List[Tuple[int, str]] = []
    # frame -> (post-replan skill texts, pointer, replanned?) for the REPLANNING hold card
    replan_info: dict = {}

    # batch accumulators
    _b_imgs:   List[torch.Tensor] = []
    _b_skills: List[str] = []

    def flush():
        if not _b_imgs:
            return
        imgs = torch.stack(_b_imgs).to(device)        # [B, T, Cams, C, H, W]
        with torch.no_grad():
            cp, bp = gate.predict(imgs, list(_b_skills))
        comp_probs_all.extend(cp.cpu().tolist())
        back_probs_all.extend(bp.cpu().tolist())
        skill_texts_all.extend(_b_skills)
        _b_imgs.clear(); _b_skills.clear()

    prev_seg_key = None
    for local_f in tqdm(range(total_frames), desc="inference"):
        f = ep_start + local_f
        # GT segment info
        seg = frame_to_seg.get(f, Segment(f, f + 1, "", -1, False))
        # Track segment changes
        if seg.start != prev_seg_key:
            seg_change_locals.append(local_f)
            seg_skills.append(seg.skill_text)
            prev_seg_key = seg.start

        # Build [T, Cams, C, H, W] history tensor
        hist_idxs = make_history_indices(f, ep_start, history_span,
                                         args.history_skip_frame, args.history_len)
        cam_seqs = []
        for idx in hist_idxs:
            data = model_ds[idx]
            cam_seqs.append(torch.stack([data[cam] for cam in args.camera_names], dim=0))
        img_seq = torch.stack(cam_seqs, dim=0)  # [T, Cams, C, H, W]

        if pipeline is not None:
            cur_local[0] = local_f        # read by remaining_cycles_fn if a recover fires below
            skill_text = pipeline.current_skill_text()
            if not ptr_skills or ptr_skills[-1] != skill_text:
                ptr_change_locals.append(local_f)
                ptr_skills.append(skill_text)
            with torch.no_grad():
                cp, bp = gate.predict(img_seq.unsqueeze(0).to(device), [skill_text])
            comp_p, back_p = float(cp.reshape(-1)[0]), float(bp.reshape(-1)[0])
            comp_probs_all.append(comp_p); back_probs_all.append(back_p)
            skill_texts_all.append(skill_text)
            plan_pos_all.append((pipeline.pointer + 1, len(pipeline.plan)))
            # The controller only decides on a tick; between ticks the last decision still stands,
            # so carry it forward rather than blanking the overlay for 14 of every 15 frames.
            action = actions_all[-1] if actions_all else "stay"
            if (not pipeline.is_done() and local_f >= history_span
                    and local_f % max(1, args.tick_stride) == 0):
                info = pipeline.step(back_prob=back_p, completion_prob=comp_p)
                action = str(info.get("action", "stay"))
                if action in ("advance", "recover"):
                    switch_locals.append((local_f, action))
                if action == "recover":
                    # Snapshot the post-replan plan so the render can show what the LLM came back
                    # with during the hold.
                    replan_info[local_f] = ([library.instruction_of(sid) for sid in pipeline.plan],
                                            pipeline.pointer, bool(info.get("replanned")))
            actions_all.append(action)
            continue
        _b_imgs.append(img_seq)
        _b_skills.append(seg.skill_text)
        if len(_b_imgs) >= args.batch_size:
            flush()
    flush()

    if pipeline is not None:
        # The headline closed-loop number: how far each pointer switch landed from the GT boundary
        # it was supposed to track. Positive = late, negative = early.
        logger.info("Closed-loop pointer switches at frames %s (skills: %s)",
                    ptr_change_locals, ptr_skills)
        if args.plan_from_library:
            logger.info("No GT segment labels in this capture — reporting switches only, with no "
                        "timing offsets (there is no boundary to measure against).")
        else:
            logger.info("GT segment boundaries at frames %s (skills: %s)",
                        seg_change_locals, seg_skills)
            for i, (pf, ps) in enumerate(zip(ptr_change_locals, ptr_skills)):
                if i < len(seg_change_locals):
                    logger.info("  switch %d -> %-40s | pointer f=%d  GT f=%d  offset %+d frames",
                                i, ps, pf, seg_change_locals[i], pf - seg_change_locals[i])
        if not pipeline.is_done():
            logger.warning("Pointer never reached the end of the plan (stuck at '%s') — the "
                           "completion head under-fires in closed loop.", pipeline.current_skill_id())
    # Convert GT back / completion frames to local indices for chart shading
    back_gt_locals = {f - ep_start for f in back_gt_abs if ep_start <= f < ep_end}
    comp_gt_locals = {f - ep_start for f in comp_gt_abs if ep_start <= f < ep_end}

    # ── 5. render + write video ────────────────────────────────────────────────
    show_back = not args.no_back_curve
    # GT overlays only mean something when the capture actually carries per-frame subtask labels.
    show_gt = not args.plan_from_library
    hold_frames = int(round(args.replan_hold_s * args.fps)) if args.replan_hold_s > 0 else 0
    n_written = 0
    if hold_frames and replan_info:
        logger.info("Replanning hold: %d recover(s) x %.1f s = %d extra frames.",
                    len(replan_info), args.replan_hold_s, len(replan_info) * hold_frames)
    logger.info("Rendering → %s", out_path)

    writer = imageio.get_writer(
        out_path, fps=args.fps, codec="libx264", macro_block_size=1,
        ffmpeg_params=["-crf", "23", "-preset", "fast"],
    )

    for local_f in tqdm(range(total_frames), desc="render"):
        f = ep_start + local_f

        # ── camera strip (left panel) ──────────────────────────────────────
        cam_strip = frame_to_bgr(raw_ds, f, args.camera_names, args.cam_h)

        # Scale camera strip to PANEL_H so left and right match in height
        ch, cw = cam_strip.shape[:2]
        scale = PANEL_H / ch
        cam_strip = cv2.resize(cam_strip, (max(1, int(cw * scale)), PANEL_H))

        # Banner: fire for BANNER_FRAMES after each pointer switch, naming the skill it switched to.
        banner = None
        if pipeline is not None:
            for sf, act in switch_locals:
                if sf <= local_f < sf + BANNER_FRAMES:
                    # skill_texts_all[sf] is the skill the pointer was on when it decided; the one it
                    # switched TO first appears on the next frame. Name that, so the banner says
                    # where the active skill went, not where it came from.
                    banner = (act, skill_texts_all[min(sf + 1, len(skill_texts_all) - 1)])
                    break

        add_overlay(
            cam_strip, args.episode, local_f, total_frames,
            skill_texts_all[local_f],
            back_probs_all[local_f],
            comp_probs_all[local_f],
            show_back=show_back,
            plan_pos=plan_pos_all[local_f] if plan_pos_all else None,
            action=actions_all[local_f] if actions_all else None,
            banner=banner,
        )

        # ── chart panel (right panel) ──────────────────────────────────────
        chart = draw_chart(
            back_probs=back_probs_all[: local_f + 1],
            comp_probs=comp_probs_all[: local_f + 1],
            seg_change_locals=seg_change_locals,
            seg_skills=seg_skills,
            back_gt_locals=back_gt_locals,
            comp_gt_locals=comp_gt_locals,
            cur_local=local_f,
            total=total_frames,
            show_back=show_back,
            tau_done=args.tau_done if pipeline is not None else None,
            switch_locals=switch_locals if pipeline is not None else None,
            win_len=max(20, args.chart_window),
            show_gt=show_gt,
        )

        composite = cv2.hconcat([cam_strip, chart])
        writer.append_data(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        n_written += 1

        # ── replanning hold: freeze on a card for the LLM round-trip a recover really costs ──
        if hold_frames and local_f in replan_info:
            plan_txt, ptr, replanned = replan_info[local_f]
            for k in range(hold_frames):
                card = draw_replan_card(composite, (k + 1) / args.fps,
                                        args.replan_hold_s, plan_txt, ptr, replanned,
                                        style=args.replan_card)
                writer.append_data(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
            n_written += hold_frames

    writer.close()
    logger.info("Done — saved %d frames @ %d fps (%d episode frames + %d held over %d replan(s)) → %s",
                n_written, args.fps, total_frames, n_written - total_frames,
                len(replan_info) if hold_frames else 0, out_path)
    print(f"\n✓ Video saved: {out_path}")


if __name__ == "__main__":
    main()
