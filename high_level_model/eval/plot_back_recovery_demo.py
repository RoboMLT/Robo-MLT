"""Publication figure: System-2 back / recovery on a blood-gas episode with human interference.

Companion to ``plot_inference_demo`` (which tells the *advance* story). This one tells the **back /
recovery** story: while the robot is processing a sample tube, a person intervenes and pulls the
sample tube away (recorded as ``back_event=1``). The gate's **back head** fires, System-2 issues a
**recover**, and the robot skips ahead to the **next sample's testing workflow** (grasp the next tube
→ remove its cap → …).

Unlike the first version of this script, the pointer here is **not** GT-synced: a real closed-loop
controller (same thresholds / debounce / cooldown semantics as :class:`System2Pipeline`) drives it
off the live gate signals, so the plotted subtask track is *what System-2 decided*, with the recorded
GT boundaries drawn only as thin reference lines. Recovery is realised the way this capture's
operator does it — jump the pointer to the next ``grasp`` in the plan — rather than through an LLM
replan, so the figure stays reproducible offline.

**Controller-view signals.** The plotted back / completion curves are the gate outputs *as the
controller consumes them*: filtered exactly as the pipeline filters them (median + EMA, reset at each
switch) and then **latched to zero once the event they encode has been acted on** — after a recover
the back event is consumed and cannot re-fire (counters reset + cooldown), and after any switch the
completion plateau that caused it is likewise consumed. Pass ``--show_raw`` to overlay the unlatched
filtered signal so the two are directly comparable.

Run from the repo root::

    python -m high_level_model.eval.plot_back_recovery_demo \
        --repo_id <path/to/data>/BloodGasAnalysis_20260726_merged \
        --gate_ckpt outputs/completion_gate_bloodgas/<run>/completion_gate_best.pth \
        --episode 1 --skill_library configs/skill_library/bloodgas.yaml \
        --f_split 2650 --tau_back 0.8 --k_b 3 --tau_done 0.6 --k_a 1 --cooldown 6 \
        --tick_stride 15 --tolerance_s 0.01 \
        --out outputs/completion_gate_bloodgas/back_fig_ep0001
"""

import argparse
import logging
import os
from typing import List, Optional, Tuple

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
from high_level_model.eval.gate_metrics import oracle_plan_from_segments
from high_level_model.planning.skill_library import load_skill_library
from high_level_model.planning.signal_filters import ProgressSignalFilter
from high_level_model.eval.plot_inference_demo import SHORT, _skill_colors, _contiguous_runs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

C_BACK = "#e87ba4"      # back probability
C_COMP = "#2a78d6"      # completion probability
C_TAU_BACK = "#c2185b"
C_TAU_DONE = "#eb6834"
C_WINDOW = "#1baf7a"    # GT valid-switch (completion) window
C_WARM = "#8a8a86"      # history warm-up band
C_GTLINE = "#8a8a86"
C_INK = "#111111"

GRASP_MARKERS = ("grasp",)   # substrings that identify a "start of a new sample" skill


def _is_grasp(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in GRASP_MARKERS)


@torch.no_grad()
def run_closed_loop(args) -> dict:
    """Closed-loop gate + pointer controller over one episode.

    The controller mirrors :meth:`System2Pipeline.step` (recover has priority over advance, both are
    debounced over *ticks*, a cooldown follows every switch, filters reset on switch), with the one
    deployment-specific substitution described in the module docstring: a recover jumps the pointer to
    the next ``grasp`` skill instead of calling the LLM replanner.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _tol = {} if args.tolerance_s is None else {"tolerance_s": args.tolerance_s}

    _pth, side = _resolve_gate_sidecar(args.gate_ckpt)
    cam_names    = side.get("camera_names") or ["observation.images.cam_top"]
    history_len  = int(side.get("history_len") or 6)
    hist_skip    = int(side.get("history_skip_frame") or 10)
    comp_window  = int(side.get("completion_window") or 50)
    comp_at_end  = bool(side.get("completion_at_episode_end", True))
    logger.info("Gate cameras=%s history_len=%d skip=%d", cam_names, history_len, hist_skip)

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
    back_gt_abs  = gate_ds.back_frames
    comp_gt_abs  = gate_ds.completion_frames
    ep_start, ep_end = _get_ep_bounds(model_ds.meta, args.episode)
    total = ep_end - ep_start

    backbone = build_siglip_encoder(device, history_len, freeze_siglip=True).to(device)
    gate = load_gate(args.gate_ckpt, backbone, device)
    history_span = history_len * hist_skip

    library = load_skill_library(args.skill_library)
    plan_ids, plan_texts = oracle_plan_from_segments(frame_to_seg, ep_start, ep_end, library)
    logger.info("Plan (%d steps): %s", len(plan_texts),
                [SHORT.get(t, t) for t in plan_texts])

    # GT segment boundaries, for reference lines only.
    seg_change: List[int] = []; seg_skills: List[str] = []; prev_key = None
    for local_f in range(total):
        seg = frame_to_seg.get(ep_start + local_f, Segment(0, 0, "", -1, False))
        if seg.start != prev_key:
            seg_change.append(local_f); seg_skills.append(seg.skill_text); prev_key = seg.start

    # ── controller state (System2Pipeline semantics) ────────────────────────────
    back_filt = ProgressSignalFilter(median_window=1, ema_alpha=0.5, monotone=False)
    comp_filt = ProgressSignalFilter(median_window=3, ema_alpha=0.5, monotone=False)
    pointer = 0
    adv_count = back_count = 0
    cooldown = 0
    back_latched = comp_latched = False    # event consumed → controller view reads 0
    back_below = comp_below = 0            # sub-threshold ticks, for releasing the latches
    b = c = 0.0                            # held between ticks when filter_rate == "tick" 

    raw_back: List[float] = []; raw_comp: List[float] = []
    view_back: List[float] = []; view_comp: List[float] = []
    filt_back: List[float] = []; filt_comp: List[float] = []
    ptr_series: List[int] = []
    advance_frames: List[int] = []; recover_frames: List[int] = []

    # Decoding dominates this loop (the gate forward barely touches the GPU). Each history window is
    # [f-50, f-40, …, f] at skip=10, so a given global frame is requested by 6 different steps spread
    # over 50 iterations — an LRU a little larger than that span turns 6 random-seek decodes per frame
    # into 1 sequential one. Without it a 5 k-frame episode takes hours.
    from functools import lru_cache

    @lru_cache(maxsize=96)
    def _cams(idx: int) -> torch.Tensor:
        data = model_ds[idx]
        return torch.stack([data[c] for c in cam_names], dim=0)   # [Cams, C, H, W]

    for local_f in tqdm(range(total), desc="closed-loop"):
        f = ep_start + local_f
        skill_text = plan_texts[pointer]

        hist_idxs = make_history_indices(f, ep_start, history_span, hist_skip, history_len)
        img_seq = torch.stack([_cams(idx) for idx in hist_idxs], dim=0)   # [T, Cams, C, H, W]
        cp, bp = gate.predict(img_seq.unsqueeze(0).to(device), [skill_text])
        b_raw, c_raw = float(bp.reshape(-1)[0]), float(cp.reshape(-1)[0])

        is_tick = local_f >= history_span and local_f % max(1, args.tick_stride) == 0
        # ``System2Pipeline.step`` — and with it the output filters — runs once per System-2 tick,
        # and deployment agrees (``System2Controller`` returns early unless
        # ``frame_counter % sampling_interval == 0``). Advancing the filters every frame instead, as
        # this script originally did, runs their EMA ~tick_stride x faster than the robot's, so the
        # value being thresholded reaches tau well before the deployed controller's would.
        # ``--filter_rate per_frame`` restores the old behaviour for comparison.
        if args.filter_rate == "per_frame" or is_tick:
            b = back_filt.update(b_raw)
            c = comp_filt.update(c_raw)
        # else: the controller's view simply does not change between ticks — b/c are held.
        switched = None
        if is_tick and pointer < len(plan_texts) - 1:
            if cooldown > 0:
                cooldown -= 1

            # Release a latch only once the signal that caused the switch has stayed decayed for a
            # full debounce window AND the cooldown has expired. Releasing on the first sub-threshold
            # sample instead lets a still-ongoing event (the tube is still missing) pop the curve back
            # up moments after it was acted on — visually a "second" event that the controller never
            # sees, because its counters were reset.
            if back_latched:
                back_below = back_below + 1 if b < args.tau_back else 0
                if back_below >= args.k_b and cooldown == 0:
                    back_latched = False
            if comp_latched:
                comp_below = comp_below + 1 if c < args.tau_done else 0
                if comp_below >= max(2, args.k_a) and cooldown == 0:
                    comp_latched = False
            back_count = back_count + 1 if (b >= args.tau_back and not back_latched) else 0
            if back_count >= args.k_b:
                nxt = next((j for j in range(pointer + 1, len(plan_texts))
                            if _is_grasp(plan_texts[j])), None)
                if nxt is not None:
                    pointer = nxt
                    switched = "recover"
                    recover_frames.append(local_f)
            elif not comp_latched and c >= args.tau_done and cooldown == 0:
                adv_count += 1
                if adv_count >= args.k_a:
                    pointer += 1
                    switched = "advance"
                    advance_frames.append(local_f)
            elif c < args.tau_done:
                adv_count = 0

            if switched:
                adv_count = back_count = 0
                cooldown = args.cooldown
                back_filt.reset(); comp_filt.reset()
                comp_latched, comp_below = True, 0        # the plateau that fired is consumed
                if switched == "recover":
                    back_latched, back_below = True, 0    # the interference event is consumed

        raw_back.append(b_raw); raw_comp.append(c_raw)
        filt_back.append(b); filt_comp.append(c)
        view_back.append(0.0 if back_latched else b)
        view_comp.append(0.0 if comp_latched else c)
        ptr_series.append(pointer)

    logger.info("Closed loop: %d advances, %d recovers, pointer ended at %d/%d",
                len(advance_frames), len(recover_frames), pointer, len(plan_texts) - 1)
    logger.info("recover frames: %s", recover_frames)

    back_gt_locals = sorted({f - ep_start for f in back_gt_abs if ep_start <= f < ep_end})
    comp_gt_locals = sorted({f - ep_start for f in comp_gt_abs if ep_start <= f < ep_end})

    # pointer-driven track: runs of a constant pointer value
    ptr_change: List[int] = []; ptr_texts: List[str] = []
    for i, p in enumerate(ptr_series):
        if i == 0 or p != ptr_series[i - 1]:
            ptr_change.append(i); ptr_texts.append(plan_texts[p])

    display_cam = f"observation.images.{args.camera}"
    key_frames = {lf: frame_to_bgr(raw_ds, ep_start + lf, [display_cam], args.cam_h)[:, :, ::-1].copy()
                  for lf in ptr_change}

    return dict(
        filter_rate=args.filter_rate,   # so replays can tell which cadence this run thresholded
        raw_back=raw_back, raw_comp=raw_comp, filt_back=filt_back, filt_comp=filt_comp,
        view_back=view_back, view_comp=view_comp,
        ptr_change=ptr_change, ptr_texts=ptr_texts, ptr_series=ptr_series,
        seg_change=seg_change, seg_skills=seg_skills,
        back_gt_locals=back_gt_locals, comp_gt_locals=comp_gt_locals,
        advance_frames=advance_frames, recover_frames=recover_frames,
        key_frames=key_frames, total=total, fps=args.fps, history_span=history_span,
        plan_texts=plan_texts, episode=args.episode,
        dataset_name=os.path.basename(os.path.normpath(args.repo_id)))


# ─────────────────────────────── rendering ──────────────────────────────────────
def _draw_track(ax, data, colors, lo, hi, fps, title=""):
    """Pointer-driven subtask track over [lo, hi) local frames."""
    ptr_change, ptr_texts = data["ptr_change"], data["ptr_texts"]
    bounds = list(ptr_change) + [data["total"]]
    for i, txt in enumerate(ptr_texts):
        a, b = bounds[i], bounds[i + 1]
        if b <= lo or a >= hi:
            continue
        a, b = max(a, lo), min(b, hi)
        ax.axvspan(a / fps, b / fps, color=colors[txt], alpha=0.9, lw=0)
        # Label only bands whose *visible* span can hold the whole word — a band clipped by the row
        # split would otherwise render a half-word at the panel edge. ~0.42 s of this axis per
        # character at fontsize 8 on the panel widths this figure uses.
        label = SHORT.get(txt, txt)
        if (b - a) / fps > 0.42 * len(label):
            ax.text((a + b) / 2 / fps, 0.5, label, ha="center", va="center",
                    fontsize=8.0, color="white", weight="bold")
    ax.set_yticks([]); ax.set_ylim(0, 1); ax.set_xlim(lo / fps, hi / fps)
    ax.set_ylabel("System-2\npointer", fontsize=8.5, rotation=0, ha="right", va="center")
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    if title:
        ax.set_title(title, fontsize=10.5, loc="left", color="#4a4a47", pad=6)


def _draw_curve(ax, data, args, lo, hi, *, show_warm: bool):
    fps = data["fps"]
    t = np.arange(data["total"]) / fps
    m = (np.arange(data["total"]) >= lo) & (np.arange(data["total"]) < hi)

    # GT valid-switch (completion) windows + GT interference window
    for a, b in _contiguous_runs(data["comp_gt_locals"]):
        ax.axvspan(a / fps, b / fps, color=C_WINDOW, alpha=0.15, lw=0, zorder=0)
    for a, b in _contiguous_runs(data["back_gt_locals"]):
        ax.axvspan(a / fps, b / fps, color=C_BACK, alpha=0.20, lw=0, zorder=0)
    if show_warm:
        warm = int(data["history_span"])
        ax.axvspan(0, warm / fps, color=C_WARM, alpha=0.16, lw=0, zorder=0)
        ax.text(warm / fps / 2, 0.50, "history\nwarm-up", ha="center", va="center", fontsize=7.2,
                color="#4a4a47", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82))

    ax.axhline(args.tau_back, ls="--", lw=1.3, color=C_TAU_BACK, zorder=2)
    ax.axhline(args.tau_done, ls="--", lw=1.3, color=C_TAU_DONE, zorder=2)
    ax.text(hi / fps - 0.4, args.tau_back + 0.03, f"τ_back = {args.tau_back:g}", ha="right",
            va="bottom", fontsize=8.5, color=C_TAU_BACK)
    # Offset past the warm-up band so the label never sits inside the shaded region.
    tau_done_x = lo / fps + 0.10 * (hi - lo) / fps
    ax.text(tau_done_x, args.tau_done + 0.03, f"τ_done = {args.tau_done:g}", ha="left",
            va="bottom", fontsize=8.5, color=C_TAU_DONE)

    if args.show_raw:
        ax.plot(t[m], np.asarray(data["filt_comp"])[m], lw=0.9, color=C_COMP, alpha=0.30, zorder=2)
        ax.plot(t[m], np.asarray(data["filt_back"])[m], lw=0.9, color=C_BACK, alpha=0.30, zorder=2)
    ax.plot(t[m], np.asarray(data["view_comp"])[m], lw=1.7, color=C_COMP, zorder=4)
    ax.plot(t[m], np.asarray(data["view_back"])[m], lw=2.0, color=C_BACK, zorder=5)

    # GT boundaries (reference only — the pointer is not synced to them)
    for lf in data["seg_change"][1:]:
        if lo <= lf < hi:
            ax.axvline(lf / fps, color=C_GTLINE, lw=0.8, alpha=0.55, zorder=1)
    # controller events
    for lf in data["advance_frames"]:
        if lo <= lf < hi:
            ax.axvline(lf / fps, color=C_INK, ls=(0, (4, 2)), lw=1.1, zorder=3)
    for lf in data["recover_frames"]:
        if lo <= lf < hi:
            ax.axvline(lf / fps, color=C_TAU_BACK, ls="-", lw=2.0, zorder=6)
            ax.annotate("recover\n(sample tube pulled away)\n→ skip to next sample",
                        xy=(lf / fps, args.tau_back), xytext=(lf / fps - 8.0, 0.46),
                        fontsize=8.2, color=C_INK, ha="right", va="center", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#d8d8d4", lw=0.7,
                                  alpha=0.94),
                        arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.1))

    ax.set_ylim(-0.03, 1.06); ax.set_xlim(lo / fps, hi / fps)
    ax.set_ylabel("gate probability", fontsize=9.5)
    ax.grid(axis="y", color="#e3e3e0", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def render_figure(data: dict, args, out_stem: str) -> Tuple[str, str]:
    fps = data["fps"]; total = data["total"]
    colors = _skill_colors(data["plan_texts"])

    f_split = args.f_split if args.f_split is not None else total // 2
    f_split = int(np.clip(f_split, 1, total - 1))

    # film strip: pointer segments inside the focus window (defaults to around the interference)
    rec = data["recover_frames"][0] if data["recover_frames"] else (
        data["back_gt_locals"][0] if data["back_gt_locals"] else 0)
    focus_lo = args.focus_lo if args.focus_lo is not None else max(0, rec - 700)
    focus_hi = args.focus_hi if args.focus_hi is not None else min(total, rec + 900)
    strip = [(lf, txt) for lf, txt in zip(data["ptr_change"], data["ptr_texts"])
             if focus_lo <= lf < focus_hi][:args.max_frames_strip]
    if not strip:
        strip = list(zip(data["ptr_change"], data["ptr_texts"]))[:args.max_frames_strip]
    n_key = len(strip)

    fig = plt.figure(figsize=(17.5, 10.2), dpi=100)
    outer = fig.add_gridspec(3, 1, height_ratios=[2.35, 3.0, 3.0], hspace=0.20,
                             left=0.062, right=0.988, top=0.895, bottom=0.105)
    strip_gs = outer[0].subgridspec(1, n_key, wspace=0.055)
    blkA = outer[1].subgridspec(2, 1, height_ratios=[0.30, 3.0], hspace=0.05)
    blkB = outer[2].subgridspec(2, 1, height_ratios=[0.30, 3.0], hspace=0.05)

    ax_curveA = fig.add_subplot(blkA[1]); ax_trackA = fig.add_subplot(blkA[0], sharex=ax_curveA)
    ax_curveB = fig.add_subplot(blkB[1]); ax_trackB = fig.add_subplot(blkB[0], sharex=ax_curveB)

    # film strip
    frame_axes = []
    for i, (lf, txt) in enumerate(strip):
        axf = fig.add_subplot(strip_gs[0, i])
        axf.imshow(data["key_frames"][lf])
        axf.set_xticks([]); axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_edgecolor(colors[txt]); sp.set_linewidth(2.4)
        tag = "  ⟲ recover" if any(abs(lf - r) <= args.tick_stride for r in data["recover_frames"]) else ""
        axf.set_title(f"{SHORT.get(txt, txt)}{tag}\nf{lf}", fontsize=9.0, color=C_INK, pad=3)
        frame_axes.append(axf)

    _draw_track(ax_trackA, data, colors, 0, f_split, fps,
                title=f"①  interference & recovery   ·   frames 0–{f_split}")
    _draw_curve(ax_curveA, data, args, 0, f_split, show_warm=True)
    _draw_track(ax_trackB, data, colors, f_split, total, fps,
                title=f"②  remaining subtasks run to completion   ·   frames {f_split}–{total}")
    _draw_curve(ax_curveB, data, args, f_split, total, show_warm=False)
    ax_curveB.set_xlabel("time (s)", fontsize=10)

    # leader lines film → track A (only for frames that fall in row A)
    for i, (lf, _txt) in enumerate(strip):
        if lf < f_split:
            fig.add_artist(ConnectionPatch(
                xyA=(0.5, 0.0), coordsA=frame_axes[i].transAxes,
                xyB=(lf / fps, 1.0), coordsB=ax_trackA.transData,
                color="#b9b9b4", lw=0.8, zorder=0))

    seen = []
    for txt in data["ptr_texts"]:
        if txt not in seen:
            seen.append(txt)
    handles = [Patch(fc=colors[s], ec="none", label=SHORT.get(s, s)) for s in seen]
    handles += [
        Line2D([0], [0], color=C_BACK, lw=2.0, label="back prob (controller view)"),
        Line2D([0], [0], color=C_COMP, lw=1.7, label="completion prob (controller view)"),
        Line2D([0], [0], color=C_TAU_BACK, ls="--", lw=1.3, label="τ_back"),
        Line2D([0], [0], color=C_TAU_DONE, ls="--", lw=1.3, label="τ_done"),
        Line2D([0], [0], color=C_TAU_BACK, lw=2.0, label="recover event"),
        Line2D([0], [0], color=C_INK, ls=(0, (4, 2)), lw=1.1, label="advance (pointer switch)"),
        Line2D([0], [0], color=C_GTLINE, lw=0.8, alpha=0.55, label="GT boundary (reference)"),
        Patch(fc=C_WINDOW, alpha=0.15, ec="none", label="GT valid-switch window"),
        Patch(fc=C_BACK, alpha=0.20, ec="none", label="GT interference (back)"),
        Patch(fc=C_WARM, alpha=0.16, ec="none", label="history warm-up"),
    ]
    if args.show_raw:
        handles.append(Line2D([0], [0], color="#9a9a98", lw=0.9, label="unlatched filtered signal"))
    ax_curveB.legend(handles=handles, ncol=6, fontsize=8.2, loc="upper center",
                     bbox_to_anchor=(0.5, -0.16), frameon=False, columnspacing=1.2,
                     handlelength=1.6)

    n_adv, n_rec = len(data["advance_frames"]), len(data["recover_frames"])
    rec_s = f"{data['recover_frames'][0] / fps:.1f}s" if data["recover_frames"] else "—"
    fig.suptitle(
        f"System-2 closed-loop back / recovery — episode {data['episode']} "
        f"({data.get('dataset_name','')})\n"
        f"a person pulls the sample tube away → back head fires (recover @ {rec_s}) → System-2 skips "
        f"to the next sample's workflow · {n_adv} gate-driven advances, {n_rec} recover · "
        f"τ_done={args.tau_done:g}, k_a={args.k_a}, τ_back={args.tau_back:g}, k_b={args.k_b}, "
        f"cooldown={args.cooldown}, tick={args.tick_stride}",
        fontsize=12.5, y=0.982)

    png, pdf = out_stem + ".png", out_stem + ".pdf"
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s and %s", png, pdf)
    return png, pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--gate_ckpt", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--skill_library", required=True)
    ap.add_argument("--camera", default="cam_top")
    ap.add_argument("--f_split", type=int, default=None,
                    help="local frame where row ① ends and row ② begins (default: episode midpoint)")
    ap.add_argument("--focus_lo", type=int, default=None, help="film-strip window start (local frame)")
    ap.add_argument("--focus_hi", type=int, default=None, help="film-strip window end (local frame)")
    ap.add_argument("--max_frames_strip", type=int, default=7)
    ap.add_argument("--tau_back", type=float, default=0.8)
    ap.add_argument("--k_b", type=int, default=3)
    ap.add_argument("--tau_done", type=float, default=0.6)
    ap.add_argument("--k_a", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=6, help="ticks blocked after a switch")
    ap.add_argument("--tick_stride", type=int, default=15)
    ap.add_argument("--filter_rate", choices=("tick", "per_frame"), default="tick",
                    help="how often to advance the gate output filters. 'tick' matches "
                         "System2Pipeline and deployment (once per tick_stride frames); "
                         "'per_frame' is this script's original behaviour, ~tick_stride x faster "
                         "than the robot.")
    ap.add_argument("--show_raw", action="store_true",
                    help="overlay the unlatched filtered signals for comparison")
    ap.add_argument("--tolerance_s", type=float, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--cam_h", type=int, default=480)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse cached inference (<out>.data.pkl)")
    args = ap.parse_args()

    out_stem = args.out or os.path.join(os.path.dirname(args.gate_ckpt.rstrip("/")),
                                        f"back_fig_ep{args.episode:04d}")
    import pickle
    cache_pkl = out_stem + ".data.pkl"
    if args.reuse and os.path.isfile(cache_pkl):
        with open(cache_pkl, "rb") as f:
            data = pickle.load(f)
        logger.info("Reusing cached inference data from %s", cache_pkl)
    else:
        data = run_closed_loop(args)
        with open(cache_pkl, "wb") as f:
            pickle.dump(data, f)
    render_figure(data, args, out_stem)


if __name__ == "__main__":
    main()
