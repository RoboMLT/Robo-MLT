"""Offline replay of the zero-shot VLM monitor on recorded episodes (no robot, no GPU).

The VLM counterpart of ``eval_switch_offline``: a :class:`System2Pipeline` pinned to the
ground-truth skill sequence (recovery off) is ticked at the deployment cadence, K control steps at
f_ctrl expressed in dataset frames, and at every tick the VLM (``planning/vlm_monitor.py``) is asked
about the pointer's CURRENT skill, so one early advance mis-conditions everything after it. The
boundaries are paired and labelled exactly as there (premature / on_time / late / missed; see
``gate_metrics.switch_verdicts``).

Latency: with ``--latency measured`` (default) an answer takes effect after the measured call
latency and no new window is sent while one is in flight, as ``System2Controller`` runs one
monitor job at a time while the recording keeps playing. ``--latency 0`` gives the instant-answer
upper bound; any other number is a fixed latency in seconds. The model's ``feasible`` answer is
logged but not acted on: on nominal episodes every "infeasible" is a would-be false recovery.

    export DASHSCOPE_API_KEY=sk-...
    python -m high_level_model.eval.eval_vlm_monitor_offline \\
        --repo_id /path/to/RoboMLT-Lab/blood_gas_analysis \\
        --skill_library configs/skill_library/bloodgas.yaml \\
        --monitor_prompt configs/skill_library/prompts/bloodgas_monitor.yaml \\
        --episodes 241 --out outputs/system2/vlm_monitor/blood_gas
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os

import numpy as np
import torch

from high_level_model.data.completion_gate_dataset import CompletionGateDataset
from high_level_model.data.high_level_dataset import _get_ep_bounds
from high_level_model.eval.gate_metrics import build_eval_pipeline, oracle_plan_from_segments, switch_verdicts
from high_level_model.planning.skill_library import SkillLibrary
from high_level_model.planning.vlm_monitor import MonitorPrompt, VLMGate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def replay(pipe, ask, first_tick: int, n: int, tick: int, cooldown: int, latency_frames):
    """Tick the pointer over one episode. ``ask(f, text) -> (gamma, beta)`` queries the window that
    ends at local frame ``f``; ``latency_frames()`` is the delay of the latest answer in frames.
    Returns (switch frames, frame the final skill completed or None, trace)."""
    switches, done_at, trace = [], None, []
    f, blocked_until = first_tick, -1
    while f < n and not pipe.is_done():
        if f < blocked_until:
            f += tick
            continue
        text = pipe.current_skill_text()
        comp, back = ask(f, text)
        ready = f + latency_frames()
        if ready >= n:                      # the answer would arrive after the recording ends
            break
        info = pipe.step(back_prob=0.0, completion_prob=comp)
        trace.append({"query": f, "ready": ready, "pointer": pipe.pointer, "completed": bool(comp),
                      "feasible": not back})
        if info["action"] == "advance":
            switches.append(ready)
            blocked_until = ready + cooldown
        elif info["action"] == "done":
            done_at = ready
        # The next window goes out on the first K-step tick at or after the answer arrived.
        f = first_tick + max(f - first_tick + tick, math.ceil((ready - first_tick) / tick) * tick)
    return switches, done_at, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True, help="LeRobot v3 dataset root")
    ap.add_argument("--skill_library", required=True)
    ap.add_argument("--monitor_prompt", required=True, help="configs/skill_library/prompts/*_monitor.yaml")
    ap.add_argument("--episodes_file", default=None, help='JSON with a "nominal" list of episode indices')
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm_model", default="qwen3.6-plus")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--latency", default="measured", help='"measured", or a fixed latency in seconds')
    ap.add_argument("--max_consecutive_errors", type=int, default=5,
                    help="abort when this many API calls in a row fail (no credit, wrong model id, ...)")
    ap.add_argument("--history_len", type=int, default=3, help="M; the VLM sees the first and the last frame")
    ap.add_argument("--sampling_interval", type=int, default=15, help="K, control steps")
    ap.add_argument("--control_fps", type=float, default=20.0)
    ap.add_argument("--cooldown_steps", type=int, default=10, help="Delta_cd, control steps")
    ap.add_argument("--tau_done", type=float, default=0.6)
    ap.add_argument("--k_a", type=int, default=2)
    ap.add_argument("--completion_window", type=int, default=50, help="frames; as the learned monitor's labels")
    ap.add_argument("--early_slack_s", type=float, default=0.5)
    ap.add_argument("--late_slack_s", type=float, default=1.5)
    ap.add_argument("--video_backend", default="pyav")
    ap.add_argument("--tolerance_s", type=float, default=1e-2)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from high_level_model.planning.llm_backends import qwen_llm_fn

    prompt = MonitorPrompt.from_file(args.monitor_prompt)
    cams = list(dict.fromkeys(c.key for c in prompt.cameras))
    dset = LeRobotDataset(repo_id=args.repo_id, root=args.repo_id, video_backend=args.video_backend,
                          tolerance_s=args.tolerance_s)
    fps = float(dset.meta.fps)
    tick = max(1, round(args.sampling_interval * fps / args.control_fps))
    cooldown = round(args.cooldown_steps * fps / args.control_fps)
    early = args.completion_window + round(args.early_slack_s * fps)
    late = round(args.late_slack_s * fps)
    first_tick = (args.history_len - 1) * tick
    if args.episodes is not None:
        episodes = list(args.episodes)
    elif args.episodes_file:
        with open(args.episodes_file) as f:
            episodes = [int(e) for e in json.load(f)["nominal"]]
    else:
        episodes = list(range(dset.meta.total_episodes))
    segs = CompletionGateDataset(dset, subset_episodes=episodes, camera_names=cams, history_len=1,
                                 history_skip_frame=1, use_command_in_meta=False,
                                 completion_window=args.completion_window, completion_at_episode_end=True)
    llm_fn = qwen_llm_fn(model=args.vlm_model, enable_thinking=False, max_tokens=args.max_tokens,
                         timeout=args.timeout)
    logger.info("VLM %s, M=%d (span %.2f s), tick=%d frames (K=%d @ %.0f Hz, data %.0f fps), cooldown=%d, "
                "latency=%s, %d episodes", args.vlm_model, args.history_len, first_tick / fps, tick,
                args.sampling_interval, args.control_fps, fps, cooldown, args.latency, len(episodes))

    os.makedirs(args.out, exist_ok=True)
    rows, episodes_out, calls = [], [], []
    for ep in episodes:
        s, e = _get_ep_bounds(dset.meta, ep)
        n = e - s
        library = SkillLibrary.from_file(args.skill_library)
        plan, plan_texts = oracle_plan_from_segments(segs.frame_to_seg, s, e, library)
        gate = VLMGate(llm_fn, prompt, library, cams, span_s=first_tick / fps)
        cache: dict[int, torch.Tensor] = {}

        def frames(f):
            if f not in cache:
                item = dset[s + f]
                cache[f] = torch.stack([item[c] for c in cams])
            return cache[f]

        def latency_frames():
            lat = gate.calls[-1]["latency_s"] if args.latency == "measured" else float(args.latency)
            return math.ceil(lat * fps)

        def ask(f, text):
            out = gate.query(frames(f - first_tick), frames(f), text)
            recent = gate.calls[-args.max_consecutive_errors:]
            if len(recent) == args.max_consecutive_errors and all(c["error"] for c in recent):
                raise RuntimeError(f"{len(recent)} VLM calls in a row failed; last: {recent[-1]['error']}")
            return out

        pipe = build_eval_pipeline(library, plan, tau_done=args.tau_done, k_a=args.k_a, cooldown=0,
                                   clock=lambda: 0.0)
        bounds = [f - s for f in range(s + 1, e) if segs.frame_to_seg[f].start != segs.frame_to_seg[f - 1].start]
        switches, done_at, trace = replay(pipe, ask, first_tick, n, tick, cooldown, latency_frames)
        for t, c in zip(trace, gate.calls):
            t.update(skill=c["skill"], latency_s=c["latency_s"], reply=c["reply"], error=c["error"])
        verdicts = switch_verdicts(switches, bounds, early, late)
        for k, (b, (verdict, off)) in enumerate(zip(bounds, verdicts)):
            rows.append({"episode": ep, "boundary": k, "from": plan_texts[k], "to": plan_texts[k + 1],
                         "gt_frame": b, "switch_frame": switches[k] if k < len(switches) else None,
                         "offset_s": None if off is None else off / fps, "verdict": verdict})
        success = (all(v == "on_time" for v, _ in verdicts) and done_at is not None
                   and len(switches) == len(bounds))
        calls += gate.calls
        episodes_out.append({"episode": ep, "frames": n, "n_boundaries": len(bounds), "switches": switches,
                             "gt_boundaries": bounds, "done_at": done_at, "final_skill_ok": done_at is not None,
                             "extra_switches": max(0, len(switches) - len(bounds)), "success": bool(success),
                             "infeasible_answers": sum(not t["feasible"] for t in trace),
                             "calls": gate.summary(), "trace": trace})
        counts = {v: sum(x == v for x, _ in verdicts) for v in ("on_time", "premature", "late", "missed")}
        logger.info("ep %d: %d boundaries, verdicts %s, done_at=%s/%d, %d calls (median %.1f s) -> %s", ep,
                    len(bounds), counts, done_at, n, len(gate.calls), gate.summary()["latency_median_s"] or 0.0,
                    "OK" if success else "FAIL")

    lat = [c["latency_s"] for c in calls]
    offsets = [r["offset_s"] for r in rows if r["offset_s"] is not None and r["verdict"] != "premature"]
    multi = [e for e in episodes_out if e["n_boundaries"] > 0]
    summary = {
        "repo_id": args.repo_id, "monitor_prompt": args.monitor_prompt, "vlm_model": args.vlm_model,
        "latency": args.latency, "history_len": args.history_len, "tick_frames": tick,
        "cooldown_frames": cooldown, "tau_done": args.tau_done, "k_a": args.k_a,
        "completion_window": args.completion_window, "early_slack_s": args.early_slack_s,
        "late_slack_s": args.late_slack_s, "episodes": len(episodes_out),
        "monitor_success_rate_multi_skill": (sum(e["success"] for e in multi) / len(multi)) if multi else None,
        "monitor_success_rate_all": sum(e["success"] for e in episodes_out) / max(1, len(episodes_out)),
        "boundaries": len(rows),
        "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in ("on_time", "premature", "late", "missed")},
        "final_skill_completion_detected": sum(e["final_skill_ok"] for e in episodes_out),
        "median_offset_s": float(np.median(offsets)) if offsets else None,
        "infeasible_answers": sum(e["infeasible_answers"] for e in episodes_out),
        "calls": len(calls), "latency_median_s": float(np.median(lat)) if lat else None,
        "tokens": int(sum(c["usage"].get("total_tokens", 0) for c in calls if c.get("usage"))),
        "unparsed": sum(not c["parsed"] for c in calls), "errors": sum(c["error"] is not None for c in calls),
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"summary": summary, "episodes": episodes_out}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "boundaries.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
