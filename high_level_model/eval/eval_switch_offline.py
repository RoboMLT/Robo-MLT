"""Offline evaluation of the System-2 monitor on recorded episodes (no robot needed).

For every requested episode the script decodes the cameras once, encodes them with the frozen
SigLIP2 tower and then reports two things:

1. **Frame-level** completion-head quality under the ground-truth skill text: average precision and
   AUROC per skill, against the training labels (last ``completion_window`` frames of a segment).
2. **Closed-loop switch timing**, the quantity that decides whether the robot works: the plan
   pointer of a :class:`System2Pipeline` (pinned to the ground-truth skill sequence, recovery off)
   is ticked at the deployment cadence (K control steps at f_ctrl, expressed in dataset frames),
   with the deployed thresholds, debounce and post-switch cooldown. The gate is always queried with
   the pointer's CURRENT skill text, so one early advance mis-conditions everything after it, as it
   would on the robot. The k-th advance is paired with the k-th ground-truth boundary and labelled

       premature  earlier than (boundary - completion_window - early_slack): the skill is not done
       on_time    within [boundary - completion_window - early_slack, boundary + late_slack]
       late       later than boundary + late_slack (but before the episode ends)
       missed     no k-th advance at all

   An episode is a *monitor success* when every boundary is on time, there is no extra advance and
   the final skill's completion is detected (its timing is reported separately: the recordings
   keep running while the arm rests at home, so the episode end is not a sharp boundary).

    python -m high_level_model.eval.eval_switch_offline \\
        --repo_id /path/to/dataset --gate_ckpt outputs/system2/<run> \\
        --skill_library configs/skill_library/bloodgas_recorded.yaml \\
        --episodes_file /path/to/heldout_test.json --out outputs/system2/<run>/offline_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from high_level_model.data.completion_gate_dataset import CompletionGateDataset
from high_level_model.data.high_level_dataset import _get_ep_bounds
from high_level_model.eval.eval_completion_gate_video import _resolve_gate_sidecar, load_gate
from high_level_model.eval.gate_metrics import build_eval_pipeline, completion_ap, oracle_plan_from_segments
from high_level_model.models.siglip_encoder import build_siglip_encoder
from high_level_model.planning.skill_library import SkillLibrary
from high_level_model.training.train_completion_gate import _FrameWindowDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def auroc(scores, labels) -> float:
    """Rank-based AUROC (NaN when one class is absent)."""
    s, y = np.asarray(scores, float), np.asarray(labels) > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def episode_features(gate, dset, frames, cams, device, workers, window=32) -> torch.Tensor:
    """Frozen SigLIP2 features ``[F, Cams, D]`` for the episode's frames (in order)."""
    # Windows of full-resolution multi-camera frames are large; pinning them fails on some drivers.
    loader = DataLoader(_FrameWindowDataset(dset, frames, cams, window), batch_size=None,
                        num_workers=workers, pin_memory=False)
    out = []
    for imgs in loader:
        feats, num_pad = gate.backbone.encode_siglip_frames(imgs.to(device).unsqueeze(0))
        out.append(feats[0, num_pad:].float())
    return torch.cat(out)


@torch.no_grad()
def gamma(gate, feats, local_frames, stride, history_len, texts, batch=256) -> np.ndarray:
    """Completion probability at each local frame, history clipped at the episode start."""
    out = []
    for i in range(0, len(local_frames), batch):
        chunk = local_frames[i:i + batch]
        idx = torch.tensor([[max(0, f - (history_len - 1 - j) * stride) for j in range(history_len)]
                            for f in chunk], device=feats.device)
        logit, _ = gate.forward_from_cached((feats[idx],), list(texts[i:i + batch]))
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True, help="LeRobot v3 dataset root")
    ap.add_argument("--gate_ckpt", required=True, help="run folder or .pth")
    ap.add_argument("--skill_library", required=True)
    ap.add_argument("--episodes_file", default=None, help='JSON with an "nominal" list of episode indices')
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampling_interval", type=int, default=15, help="K, control steps")
    ap.add_argument("--control_fps", type=float, default=20.0)
    ap.add_argument("--cooldown_steps", type=int, default=10, help="Delta_cd, control steps")
    ap.add_argument("--tau_done", type=float, default=0.6)
    ap.add_argument("--k_a", type=int, default=2)
    ap.add_argument("--early_slack_s", type=float, default=0.5)
    ap.add_argument("--late_slack_s", type=float, default=1.5)
    ap.add_argument("--dense_stride", type=int, default=2, help="frame stride of the frame-level metrics")
    ap.add_argument("--video_backend", default="pyav")
    ap.add_argument("--tolerance_s", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _, side = _resolve_gate_sidecar(args.gate_ckpt)
    cams = list(side["camera_names"])
    history_len = int(side.get("history_len") or 1)
    stride = int(side.get("history_skip_frame") or 1)
    window = int(side.get("completion_window") or 0)
    tf = [T.ToDtype(torch.float32, scale=True)]
    if side.get("imagenet_normalize", True):
        tf.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    dset = LeRobotDataset(repo_id=args.repo_id, image_transforms=T.Compose(tf),
                          video_backend=args.video_backend, tolerance_s=args.tolerance_s)
    fps = float(dset.meta.fps)
    tick = max(1, round(args.sampling_interval * fps / args.control_fps))
    cooldown = round(args.cooldown_steps * fps / args.control_fps)
    early = window + round(args.early_slack_s * fps)
    late = round(args.late_slack_s * fps)

    if args.episodes is not None:
        episodes = list(args.episodes)
    elif args.episodes_file:
        with open(args.episodes_file) as f:
            episodes = [int(e) for e in json.load(f)["nominal"]]
    else:
        episodes = list(range(dset.meta.total_episodes))
    gate_ds = CompletionGateDataset(dset, subset_episodes=episodes, camera_names=cams, history_len=1,
                                    history_skip_frame=1, use_command_in_meta=False,
                                    completion_window=window,
                                    completion_at_episode_end=bool(side.get("completion_at_episode_end", True)),
                                    completion_label_column=side.get("completion_label_column"))
    backbone = build_siglip_encoder(device, history_len, freeze_siglip=True).to(device)
    gate = load_gate(args.gate_ckpt, backbone, device).eval()
    logger.info("M=%d stride=%d frames, tick=%d frames (K=%d @ %.0f Hz, data %.0f fps), cooldown=%d, "
                "window=%d, %d episodes", history_len, stride, tick, args.sampling_interval,
                args.control_fps, fps, cooldown, window, len(episodes))

    os.makedirs(args.out, exist_ok=True)
    dense = defaultdict(lambda: ([], []))          # skill text -> (probs, labels)
    rows, episodes_out = [], []
    for ep in episodes:
        s, e = _get_ep_bounds(dset.meta, ep)
        n = e - s
        if n <= (history_len - 1) * stride + tick:
            logger.warning("episode %d too short (%d frames) for M=%d; skipped", ep, n, history_len)
            continue
        feats = episode_features(gate, dset, list(range(s, e)), cams, device, args.workers)
        segs = [gate_ds.frame_to_seg.get(f) for f in range(s, e)]
        # (1) frame-level, GT skill text
        loc = list(range((history_len - 1) * stride, n, args.dense_stride))
        probs = gamma(gate, feats, loc, stride, history_len, [segs[f].skill_text for f in loc])
        for f, p in zip(loc, probs):
            dense[segs[f].skill_text][0].append(float(p))
            dense[segs[f].skill_text][1].append(int((s + f) in gate_ds.completion_frames))
        # (2) closed loop at the deployment cadence
        library = SkillLibrary.from_file(args.skill_library)
        plan, plan_texts = oracle_plan_from_segments(gate_ds.frame_to_seg, s, e, library)
        pipe = build_eval_pipeline(library, plan, tau_done=args.tau_done, k_a=args.k_a, cooldown=0,
                                   clock=lambda: 0.0)
        bounds = [f - s for f in range(s + 1, e) if gate_ds.frame_to_seg[f].start != gate_ds.frame_to_seg[f - 1].start]
        switches, done_at, blocked_until, trace = [], None, -1, []
        for f in range((history_len - 1) * stride, n, tick):
            if pipe.is_done():
                break
            if f < blocked_until:
                continue
            text = pipe.current_skill_text()
            p = float(gamma(gate, feats, [f], stride, history_len, [text])[0])
            info = pipe.step(back_prob=0.0, completion_prob=p)
            trace.append((f, pipe.pointer, round(p, 4)))
            if info["action"] == "advance":
                switches.append(f)
                blocked_until = f + cooldown
            elif info["action"] == "done":
                done_at = f
        verdicts = []
        for k, b in enumerate(bounds):
            if k >= len(switches):
                verdict, off = "missed", None
            else:
                off = switches[k] - b
                verdict = "premature" if off < -early else ("late" if off > late else "on_time")
            verdicts.append(verdict)
            rows.append({"episode": ep, "boundary": k, "from": plan_texts[k], "to": plan_texts[k + 1],
                         "gt_frame": b, "switch_frame": None if k >= len(switches) else switches[k],
                         "offset_s": None if off is None else off / fps, "verdict": verdict})
        # The last segment ends with the recording, which often keeps running while the arm rests at
        # home, so its end is not a sharp boundary: require only that completion is detected.
        final_ok = done_at is not None
        success = all(v == "on_time" for v in verdicts) and final_ok and len(switches) == len(bounds)
        episodes_out.append({"episode": ep, "frames": n, "n_boundaries": len(bounds), "switches": switches,
                             "gt_boundaries": bounds, "done_at": done_at, "final_skill_ok": bool(final_ok),
                             "final_detected_before_end_s": None if done_at is None else (n - done_at) / fps,
                             "extra_switches": max(0, len(switches) - len(bounds)), "success": bool(success),
                             "trace": trace})
        logger.info("ep %d: %d boundaries, verdicts %s, done_at=%s/%d -> %s", ep, len(bounds),
                    dict(zip(*np.unique(verdicts, return_counts=True))) if verdicts else {}, done_at, n,
                    "OK" if success else "FAIL")

    per_skill = {t: {"ap": completion_ap(p, y), "auroc": auroc(p, y), "frames": len(y), "positives": int(sum(y))}
                 for t, (p, y) in dense.items()}
    all_p = [x for p, _ in dense.values() for x in p]
    all_y = [x for _, y in dense.values() for x in y]
    by_boundary = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_boundary[f"{r['from']} -> {r['to']}"][r["verdict"]] += 1
    offsets = [r["offset_s"] for r in rows if r["offset_s"] is not None and r["verdict"] != "premature"]
    multi = [e for e in episodes_out if e["n_boundaries"] > 0]
    summary = {
        "gate_ckpt": args.gate_ckpt, "repo_id": args.repo_id, "history_len": history_len,
        "tick_frames": tick, "cooldown_frames": cooldown, "tau_done": args.tau_done, "k_a": args.k_a,
        "completion_window": window, "early_slack_s": args.early_slack_s, "late_slack_s": args.late_slack_s,
        "episodes": len(episodes_out),
        "monitor_success_rate_multi_skill": (sum(e["success"] for e in multi) / len(multi)) if multi else None,
        "monitor_success_rate_all": sum(e["success"] for e in episodes_out) / max(1, len(episodes_out)),
        "boundaries": len(rows),
        "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in ("on_time", "premature", "late", "missed")},
        "final_skill_completion_detected": sum(e["final_skill_ok"] for e in episodes_out),
        "final_detected_before_end_s_median": float(np.median([e["final_detected_before_end_s"] for e in episodes_out
                                                              if e["final_detected_before_end_s"] is not None]))
        if any(e["final_detected_before_end_s"] is not None for e in episodes_out) else None,
        "median_offset_s": float(np.median(offsets)) if offsets else None,
        "frame_ap": completion_ap(all_p, all_y), "frame_auroc": auroc(all_p, all_y),
        "per_skill": per_skill, "per_boundary": {k: dict(v) for k, v in by_boundary.items()},
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"summary": summary, "episodes": episodes_out}, f, indent=2)
    with open(os.path.join(args.out, "boundaries.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_skill",)}, indent=2))
    print(json.dumps(per_skill, indent=2))


if __name__ == "__main__":
    main()
