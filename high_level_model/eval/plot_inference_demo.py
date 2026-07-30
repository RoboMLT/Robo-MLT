"""Publication figure: System-2 closed-loop subtask advancement on a held-out sorting episode.

Renders ONE static figure (PNG @300dpi + vector PDF) that tells the deployment story of the
completion gate on a single episode:

  * a film strip of key node frames (cam_top) — the robot's state at each subtask transition;
  * a subtask track — the ground-truth skill sequence as a coloured timeline;
  * the completion-probability curve the gate produces under the CURRENT pointer skill, with the
    decision threshold, the ground-truth "valid switch windows", the true segment boundaries, and
    the frame at which the pointer actually advanced (annotated with its offset).

It reuses the exact closed-loop inference path of ``eval_completion_gate_video`` (same history
warm-up guard, same tick_stride gating, same oracle plan + recovery-disabled pipeline), so the
numbers in the figure match the video / sweep rescore for the same operating point.

Run from the repo root::

    python -m high_level_model.eval.plot_inference_demo \
        --repo_id /path/TubeSort_20260720 \
        --gate_ckpt outputs/sweep_20260720/A_baseline_s0/20260721/A_baseline_s0 \
        --episode 189 --skill_library configs/skill_library/bloodtube_sorting.yaml \
        --camera cam_top --tau_done 0.6 --k_a 1 --cooldown 10 --tick_stride 15 \
        --tolerance_s 0.01 --out outputs/sweep_20260720/paper_fig_ep189
"""

import argparse
import logging
import os
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from torchvision.transforms import v2 as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, ConnectionPatch

from high_level_model.data.completion_gate_dataset import CompletionGateDataset, Segment
from high_level_model.data.high_level_dataset import _get_ep_bounds
from high_level_model.models.siglip_encoder import build_siglip_encoder
from high_level_model.eval.eval_completion_gate_video import (
    _resolve_gate_sidecar, load_gate, make_history_indices, frame_to_bgr)
from high_level_model.eval.gate_metrics import build_eval_pipeline, oracle_plan_from_segments
from high_level_model.planning.skill_library import load_skill_library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Short, paper-friendly labels for the on-figure skill annotations (full text stays in the data).
# Covers both the TubeSort_20260720 sorting skills and the BloodGasAnalysis_20260722 skills; an
# unlisted skill simply falls back to its full instruction text.
SHORT = {
    # TubeSort_20260720
    "grasp the routine blood tube.":       "grasp routine",
    "grasp the sodium citrate tube.":      "grasp sodium",
    "drop the tube into the orange plate.": "→ orange",
    "place to the blue plate.":            "→ blue",
    "return to the home position.":        "home",
    # BloodGasAnalysis_20260722
    "remove the sample tube cap.":                                   "remove cap",
    "discard the tube cap into the waste bin.":                      "discard cap",
    "dock the blood gas sample tube to the blood gas analyzer.":     "dock analyzer",
    "insert the sample tube into the analyzer probe.":              "insert probe",
    "withdraw the blood gas sample tube after testing is complete.": "withdraw tube",
    "discard the tested blood sample into the waste bin.":           "discard sample",
    "return the robotic arm to the home position.":                 "home",
}
# Validated categorical palette (dataviz reference), assigned to skills in a fixed order so a skill
# keeps its colour wherever it appears. Eight distinct hues (BloodGas has 8 subtasks). Identity is
# never colour-alone — every band and frame is also text-labelled.
# Slot 8 is #b3541e, not the matplotlib-default brown #8c564b: the latter fails the dataviz chroma
# floor (OKLCh C=0.075 < 0.10, i.e. it reads as gray rather than as a hue). The one remaining
# validator WARN — #1baf7a (green) vs #e87ba4 (pink), deutan ΔE 6.1, inside the 6–8 floor band — is
# admissible here only because identity is never colour-alone: every track band carries its skill
# name in-band and every film frame is titled.
PALETTE = ["#2a78d6", "#eda100", "#e87ba4", "#1baf7a",
           "#8a63d2", "#eb6834", "#17a2b8", "#b3541e"]


def _skill_colors(skill_texts_in_order: List[str]) -> dict:
    """Deterministic skill→colour, ordered by first appearance so the legend reads left-to-right."""
    uniq: List[str] = []
    for s in skill_texts_in_order:
        if s not in uniq:
            uniq.append(s)
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(uniq)}


def _contiguous_runs(locals_sorted: List[int]) -> List[Tuple[int, int]]:
    """[3,4,5,9,10] → [(3,5),(9,10)]. Used to shade the GT completion (valid-switch) windows."""
    runs: List[Tuple[int, int]] = []
    if not locals_sorted:
        return runs
    start = prev = locals_sorted[0]
    for x in locals_sorted[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    return runs


@torch.no_grad()
def run_closed_loop(args) -> dict:
    """Closed-loop inference over one episode. Mirrors eval_completion_gate_video's loop exactly."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _tol = {} if args.tolerance_s is None else {"tolerance_s": args.tolerance_s}

    _pth, side = _resolve_gate_sidecar(args.gate_ckpt)
    cam_names   = side.get("camera_names") or ["observation.images.cam_top"]
    history_len = int(side.get("history_len") or 6)
    hist_skip   = int(side.get("history_skip_frame") or 10)
    comp_window  = int(side.get("completion_window") or 50)
    comp_at_end  = bool(side.get("completion_at_episode_end", True))
    logger.info("Gate cameras=%s history_len=%d skip=%d completion_window=%d",
                cam_names, history_len, hist_skip, comp_window)

    norm = T.Compose([T.ToDtype(torch.float32, scale=True),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model_ds = LeRobotDataset(repo_id=args.repo_id, image_transforms=norm, **_tol)
    raw_ds = LeRobotDataset(repo_id=args.repo_id,
                            image_transforms=T.Compose([T.ToDtype(torch.float32, scale=True)]), **_tol)

    gate_ds = CompletionGateDataset(
        model_ds, subset_episodes=[args.episode], camera_names=cam_names,
        history_len=history_len, prediction_offset=0, history_skip_frame=hist_skip,
        use_command_in_meta=False,
        completion_window=comp_window, completion_at_episode_end=comp_at_end)
    task_instruction = gate_ds.get_task_instruction(args.episode)
    frame_to_seg = gate_ds.frame_to_seg
    comp_gt_abs  = gate_ds.completion_frames
    ep_start, ep_end = _get_ep_bounds(model_ds.meta, args.episode)
    total = ep_end - ep_start

    backbone = build_siglip_encoder(device, history_len, freeze_siglip=True).to(device)
    gate = load_gate(args.gate_ckpt, backbone, device)
    history_span = history_len * hist_skip

    library = load_skill_library(args.skill_library)
    plan_ids, plan_texts = oracle_plan_from_segments(frame_to_seg, ep_start, ep_end, library)
    pipe = build_eval_pipeline(library, plan_ids, task_instruction,
                               tau_done=args.tau_done, k_a=args.k_a, cooldown=args.cooldown)
    pipe.gate = gate
    logger.info("Closed-loop: %d-step oracle plan, op tau=%.2f k_a=%d cd=%d tick_stride=%d",
                len(plan_ids), args.tau_done, args.k_a, args.cooldown, args.tick_stride)

    comp_probs: List[float] = []
    seg_change: List[int] = []; seg_skills: List[str] = []; prev_key = None
    ptr_change: List[int] = []; ptr_skills: List[str] = []

    # Each step needs a `history_len`-frame window, so without memoisation every global frame is
    # re-decoded once per window it appears in — six random video seeks per step, which puts this
    # loop at seconds per frame (hours per episode) and leaves the GPU idle. Caching the per-index
    # fetch makes the decode sequential and one-shot; the window is short, so a small cache holds it.
    @lru_cache(maxsize=96)
    def _cams(idx: int) -> torch.Tensor:
        data = model_ds[idx]
        return torch.stack([data[c] for c in cam_names], dim=0)      # [Cams, C, H, W]

    for local_f in range(total):
        f = ep_start + local_f
        seg = frame_to_seg.get(f, Segment(f, f + 1, "", -1, False))
        if seg.start != prev_key:
            seg_change.append(local_f); seg_skills.append(seg.skill_text); prev_key = seg.start

        hist_idxs = make_history_indices(f, ep_start, history_span, hist_skip, history_len)
        img_seq = torch.stack([_cams(idx) for idx in hist_idxs], dim=0)   # [T, Cams, C, H, W]

        skill_text = pipe.current_skill_text()
        if not ptr_skills or ptr_skills[-1] != skill_text:
            ptr_change.append(local_f); ptr_skills.append(skill_text)

        cp, bp = gate.predict(img_seq.unsqueeze(0).to(device), [skill_text])
        comp_p, back_p = float(cp.reshape(-1)[0]), float(bp.reshape(-1)[0])
        comp_probs.append(comp_p)
        if (not pipe.is_done() and local_f >= history_span
                and local_f % max(1, args.tick_stride) == 0):
            pipe.step(back_prob=back_p, completion_prob=comp_p)

    comp_gt_locals = sorted({f - ep_start for f in comp_gt_abs if ep_start <= f < ep_end})

    # Log the switch/boundary table (identical shape to the video eval) for cross-checking.
    logger.info("Pointer switches (local frames): %s", ptr_change)
    logger.info("GT boundaries    (local frames): %s", seg_change)
    for i, pf in enumerate(ptr_change):
        if i < len(seg_change):
            logger.info("  switch %d -> %-14s | pointer f=%d GT f=%d offset %+d",
                        i, SHORT.get(ptr_skills[i], ptr_skills[i]), pf, seg_change[i],
                        pf - seg_change[i])

    # Key node frame (cam_top) at each GT segment start.
    display_cam = f"observation.images.{args.camera}"
    key_frames = []
    for lf in seg_change:
        bgr = frame_to_bgr(raw_ds, ep_start + lf, [display_cam], args.cam_h)
        key_frames.append(bgr[:, :, ::-1].copy())      # BGR → RGB

    return dict(
        comp_probs=comp_probs, seg_change=seg_change, seg_skills=seg_skills,
        ptr_change=ptr_change, ptr_skills=ptr_skills, comp_gt_locals=comp_gt_locals,
        key_frames=key_frames, total=total, fps=args.fps, tau=args.tau_done,
        episode=args.episode, comp_window=comp_window, history_span=history_span,
        dataset_name=os.path.basename(os.path.normpath(args.repo_id)))


def _smooth_completion(comp: List[float], reset_at: List[int],
                       median_window: int = 3, ema_alpha: float = 0.5) -> np.ndarray:
    """Reproduce the pipeline's completion filter for display: median-window + EMA, reset at each
    pointer advance (System2Pipeline resets ``_comp_filter`` on advance). This is the signal the
    controller actually thresholds, so the plotted curve then matches where the switches fire."""
    from high_level_model.planning.signal_filters import ProgressSignalFilter
    filt = ProgressSignalFilter(median_window=median_window, ema_alpha=ema_alpha, monotone=False)
    resets = set(int(r) for r in reset_at)
    out = []
    for i, v in enumerate(comp):
        if i in resets:
            filt.reset()
        out.append(filt.update(float(v)))
    return np.asarray(out)


def render_figure(data: dict, out_stem: str) -> Tuple[str, str]:
    fps = data["fps"]
    total = data["total"]
    t = np.arange(total) / fps
    seg_change = data["seg_change"]; seg_skills = data["seg_skills"]
    ptr_change = data["ptr_change"]
    tau = data["tau"]
    colors = _skill_colors(seg_skills)
    n_key = len(seg_change)
    # History warm-up: the first history_span frames feed a padded (repeated-first-frame) window, so
    # the gate output there is meaningless (it saturates to ~1.0) and the pipeline does not even step.
    # Shade it and start the curve after it, so the plotted signal begins with real history (≈0).
    warm = int(data.get("history_span", 60))
    # Smooth the completion signal exactly as the pipeline does before thresholding (reset at each
    # pointer advance), so the curve is the signal the controller sees and matches the switch marks.
    comp = _smooth_completion(data["comp_probs"], reset_at=ptr_change)

    # ── layout: film strip (row 0) | subtask track (row 1) | completion curve (row 2) ──
    fig = plt.figure(figsize=(15, 7.4), dpi=100)
    gs = fig.add_gridspec(3, n_key, height_ratios=[2.5, 0.34, 3.1],
                          hspace=0.06, wspace=0.08, left=0.055, right=0.985,
                          top=0.9, bottom=0.11)

    # subtask-track + curve axes share the time x-axis
    ax_curve = fig.add_subplot(gs[2, :])
    ax_track = fig.add_subplot(gs[1, :], sharex=ax_curve)

    seg_bounds_t = [lf / fps for lf in seg_change] + [total / fps]

    # ---- film strip: one cam_top frame per subtask segment ----
    frame_axes = []
    for i, img in enumerate(data["key_frames"]):
        axf = fig.add_subplot(gs[0, i])
        axf.imshow(img)
        axf.set_xticks([]); axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_edgecolor(colors[seg_skills[i]]); sp.set_linewidth(2.4)
        axf.set_title(f"{SHORT.get(seg_skills[i], seg_skills[i])}\nf{seg_change[i]}",
                      fontsize=9.5, color="#111", pad=3)
        frame_axes.append(axf)

    # ---- subtask track: GT skill sequence as a coloured timeline ----
    for i in range(n_key):
        t0, t1 = seg_bounds_t[i], seg_bounds_t[i + 1]
        ax_track.axvspan(t0, t1, color=colors[seg_skills[i]], alpha=0.85, lw=0)
        ax_track.text((t0 + t1) / 2, 0.5, SHORT.get(seg_skills[i], seg_skills[i]),
                      ha="center", va="center", fontsize=8.5, color="white", weight="bold")
    ax_track.set_yticks([]); ax_track.set_ylim(0, 1)
    ax_track.set_ylabel("subtask\n(GT)", fontsize=9, rotation=0, ha="right", va="center")
    for sp in ax_track.spines.values():
        sp.set_visible(False)
    plt.setp(ax_track.get_xticklabels(), visible=False)

    # ---- completion curve panel ----
    # valid-switch windows (last completion_window frames before each true boundary)
    for a, b in _contiguous_runs(data["comp_gt_locals"]):
        ax_curve.axvspan(a / fps, b / fps, color="#1baf7a", alpha=0.16, lw=0, zorder=0)
    # history warm-up band (padded history → gate output not meaningful here)
    ax_curve.axvspan(0, warm / fps, color="#8a8a86", alpha=0.13, lw=0, zorder=0)
    ax_curve.text(warm / fps / 2, 1.0, "history\nwarm-up", ha="center", va="top", fontsize=7.5,
                  color="#6a6a66", zorder=5)
    # threshold
    ax_curve.axhline(tau, ls="--", lw=1.4, color="#eb6834", zorder=2)
    ax_curve.text(total / fps * 0.998, tau + 0.02, f"τ_done = {tau:g}", ha="right", va="bottom",
                  fontsize=9, color="#eb6834")
    # completion probability (smoothed as the controller sees it), plotted only past warm-up
    m = t >= warm / fps
    ax_curve.plot(t[m], comp[m], lw=1.8, color="#2a78d6", zorder=4, label="completion prob")
    # GT boundaries (solid grey verticals) — skip the episode start
    for lf in seg_change[1:]:
        ax_curve.axvline(lf / fps, color="#8a8a86", lw=1.1, zorder=1)
    # pointer switches (dashed) + offset annotation
    for i, pf in enumerate(ptr_change):
        if i == 0:
            continue
        off = pf - seg_change[i] if i < len(seg_change) else None
        ax_curve.axvline(pf / fps, color="#111", ls=(0, (4, 2)), lw=1.3, zorder=3)
        if off is not None:
            ax_curve.annotate(f"{off:+d}f", xy=(pf / fps, 0.06), fontsize=8,
                              color="#111", ha="center", va="bottom",
                              bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    ax_curve.set_ylim(-0.02, 1.05)
    ax_curve.set_xlim(0, total / fps)
    ax_curve.set_xlabel("time (s)", fontsize=10)
    ax_curve.set_ylabel("completion probability", fontsize=10)
    ax_curve.grid(axis="y", color="#e3e3e0", lw=0.7)
    for sp in ("top", "right"):
        ax_curve.spines[sp].set_visible(False)

    # leader lines: each film frame → its boundary on the subtask track
    for i, axf in enumerate(frame_axes):
        con = ConnectionPatch(
            xyA=(0.5, 0.0), coordsA=axf.transAxes,
            xyB=(seg_change[i] / fps, 1.0), coordsB=ax_track.transData,
            color="#b9b9b4", lw=0.8, zorder=0)
        fig.add_artist(con)

    # legend
    handles = [Patch(fc=colors[s], ec="none", label=SHORT.get(s, s))
               for s in _skill_colors(seg_skills)]
    handles += [
        Line2D([0], [0], color="#2a78d6", lw=1.8, label="completion prob (median-3 + EMA)"),
        Line2D([0], [0], color="#eb6834", ls="--", lw=1.4, label="τ_done"),
        Line2D([0], [0], color="#8a8a86", lw=1.1, label="GT boundary"),
        Line2D([0], [0], color="#111", ls=(0, (4, 2)), lw=1.3, label="pointer switch"),
        Patch(fc="#1baf7a", alpha=0.16, ec="none", label="valid-switch window"),
        Patch(fc="#8a8a86", alpha=0.13, ec="none", label="history warm-up"),
    ]
    ax_curve.legend(handles=handles, ncol=6, fontsize=8.3, loc="upper center",
                    bbox_to_anchor=(0.5, -0.13), frameon=False, columnspacing=1.1,
                    handlelength=1.5)

    matched = [pf - seg_change[i] for i, pf in enumerate(ptr_change)
               if 0 < i < len(seg_change)]
    early = sorted(-o for o in matched) if matched else [0]      # +ve = frames BEFORE the boundary
    e_lo, e_hi = early[0], early[-1]
    fig.suptitle(
        f"System-2 closed-loop subtask advancement — held-out episode {data['episode']} "
        f"({data.get('dataset_name', '')})\n"
        f"all {len(seg_change) - 1} transitions matched · pointer advances "
        f"{e_lo}–{e_hi} frames ({e_lo / fps:.2f}–{e_hi / fps:.2f} s) before the true boundary · "
        f"operating point τ={tau:g}, k_a={data.get('k_a', 1)}, tick=15",
        fontsize=12.5, y=0.985)

    png = out_stem + ".png"
    pdf = out_stem + ".pdf"
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s and %s", png, pdf)
    return png, pdf


def render_from_final(final_path: str, out_stem: str, fps: int = 25) -> Tuple[str, str]:
    """Curve-only figure from a sweep ``final.json`` (no dataset / no video needed).

    Fallback for when the raw dataset is unavailable (e.g. disk failure): the sweep already saved,
    per run, the closed-loop ``switch`` report (GT boundaries, pointer switches, offsets, the oracle
    plan skills) and the per-frame ``trace`` (completion probability + the GT valid-switch mask). That
    is the whole timeline except the camera key-frames, which are the only part that needs the video.
    """
    import json
    with open(final_path) as f:
        final = json.load(f)
    sw = final["switch"]; tr = final["trace"]
    xs = np.asarray(tr["local_frames"], dtype=float)
    comp = np.asarray(tr["completion_prob"], dtype=float)
    comp_gt = np.asarray(tr["completion_gt"], dtype=float)
    t = xs / fps
    gt_bounds = sw["gt_boundaries"]
    switches = sw["switch_frames"]
    offsets = sw.get("offsets", [])
    plan_texts = sw["plan_texts"]
    tau = float(sw.get("operating_point", {}).get("tau_done", 0.6))
    k_a = int(sw.get("operating_point", {}).get("k_a", 1))
    total = int(xs.max()) + 1
    colors = _skill_colors(plan_texts)

    seg_bounds = [0] + list(gt_bounds) + [total]          # 9 segments for 8 boundaries
    fig = plt.figure(figsize=(13.5, 4.6), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.34, 3.0], hspace=0.08,
                          left=0.075, right=0.985, top=0.82, bottom=0.2)
    ax_curve = fig.add_subplot(gs[1, 0])
    ax_track = fig.add_subplot(gs[0, 0], sharex=ax_curve)

    # subtask track (GT skill sequence)
    for i in range(len(plan_texts)):
        t0, t1 = seg_bounds[i] / fps, seg_bounds[i + 1] / fps
        ax_track.axvspan(t0, t1, color=colors[plan_texts[i]], alpha=0.85, lw=0)
        ax_track.text((t0 + t1) / 2, 0.5, SHORT.get(plan_texts[i], plan_texts[i]),
                      ha="center", va="center", fontsize=8.2, color="white", weight="bold")
    ax_track.set_yticks([]); ax_track.set_ylim(0, 1)
    ax_track.set_ylabel("subtask\n(GT)", fontsize=9, rotation=0, ha="right", va="center")
    for sp in ax_track.spines.values():
        sp.set_visible(False)
    plt.setp(ax_track.get_xticklabels(), visible=False)

    # valid-switch windows
    mask_locals = [int(x) for x, g in zip(xs, comp_gt) if g > 0.5]
    for a, b in _contiguous_runs(sorted(mask_locals)):
        ax_curve.axvspan(a / fps, b / fps, color="#1baf7a", alpha=0.16, lw=0, zorder=0)
    ax_curve.axhline(tau, ls="--", lw=1.4, color="#eb6834", zorder=2)
    ax_curve.text(total / fps * 0.998, tau + 0.02, f"τ_done = {tau:g}", ha="right", va="bottom",
                  fontsize=9, color="#eb6834")
    ax_curve.plot(t, comp, lw=1.6, color="#2a78d6", zorder=4)
    for b in gt_bounds:
        ax_curve.axvline(b / fps, color="#8a8a86", lw=1.1, zorder=1)
    for i, pf in enumerate(switches):
        ax_curve.axvline(pf / fps, color="#111", ls=(0, (4, 2)), lw=1.3, zorder=3)
        if i < len(offsets):
            ax_curve.annotate(f"{offsets[i]:+d}f", xy=(pf / fps, 0.06), fontsize=8, color="#111",
                              ha="center", va="bottom",
                              bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    ax_curve.set_ylim(-0.02, 1.05); ax_curve.set_xlim(0, total / fps)
    ax_curve.set_xlabel("time (s)", fontsize=10)
    ax_curve.set_ylabel("completion probability", fontsize=10)
    ax_curve.grid(axis="y", color="#e3e3e0", lw=0.7)
    for sp in ("top", "right"):
        ax_curve.spines[sp].set_visible(False)

    handles = [Patch(fc=colors[s], ec="none", label=SHORT.get(s, s)) for s in _skill_colors(plan_texts)]
    handles += [
        Line2D([0], [0], color="#2a78d6", lw=1.6, label="completion prob (open-loop)"),
        Line2D([0], [0], color="#eb6834", ls="--", lw=1.4, label="τ_done"),
        Line2D([0], [0], color="#8a8a86", lw=1.1, label="GT boundary"),
        Line2D([0], [0], color="#111", ls=(0, (4, 2)), lw=1.3, label="pointer switch (closed-loop)"),
        Patch(fc="#1baf7a", alpha=0.16, ec="none", label="valid-switch window"),
    ]
    ax_curve.legend(handles=handles, ncol=4, fontsize=8.2, loc="upper center",
                    bbox_to_anchor=(0.5, -0.2), frameon=False, columnspacing=1.1, handlelength=1.5)

    matched = [o for o in offsets]
    lo, hi = (min(matched), max(matched)) if matched else (0, 0)
    fig.suptitle(
        f"System-2 closed-loop subtask advancement — held-out episode {sw.get('episode', '?')} "
        f"(TubeSort_20260720, {final.get('variant', '')})\n"
        f"all {len(gt_bounds)} transitions matched · switches land {lo:+d}…{hi:+d} frames "
        f"({lo/fps:+.2f}…{hi/fps:+.2f} s) around the true boundary · τ={tau:g}, k_a={k_a}, tick=15",
        fontsize=11.5, y=0.99)

    png, pdf = out_stem + ".png", out_stem + ".pdf"
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s and %s (curve-only, from %s)", png, pdf, final_path)
    return png, pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--gate_ckpt", required=True, help="run folder or .pth")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--skill_library", required=True)
    ap.add_argument("--camera", default="cam_top",
                    help="camera SUFFIX for the film-strip frames (observation.images.<camera>)")
    ap.add_argument("--tau_done", type=float, default=0.6)
    ap.add_argument("--k_a", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--tick_stride", type=int, default=15)
    ap.add_argument("--tolerance_s", type=float, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--cam_h", type=int, default=480)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None, help="output stem (no extension); writes .png + .pdf")
    ap.add_argument("--from_final", default=None,
                    help="skip inference; render a curve-only figure from a sweep final.json "
                         "(no dataset/video needed)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the cached inference data (<out>.data.pkl) instead of recomputing — "
                         "for fast re-rendering after a cosmetic change")
    args = ap.parse_args()

    out_stem = args.out or os.path.join(os.path.dirname(args.gate_ckpt.rstrip("/")),
                                        f"paper_fig_ep{args.episode:04d}")
    if args.from_final:
        render_from_final(args.from_final, out_stem, fps=args.fps)
        return

    import pickle
    cache_pkl = out_stem + ".data.pkl"
    if args.reuse and os.path.isfile(cache_pkl):
        with open(cache_pkl, "rb") as f:
            data = pickle.load(f)
        logger.info("Reusing cached inference data from %s", cache_pkl)
    else:
        data = run_closed_loop(args)
        data["k_a"] = args.k_a
        with open(cache_pkl, "wb") as f:
            pickle.dump(data, f)
    render_figure(data, out_stem)


if __name__ == "__main__":
    main()
