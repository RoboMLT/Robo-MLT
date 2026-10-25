"""Planner check on recorded captures: plan from each episode's first global-camera frame I_0.

Every recorded blood-gas workflow holds a single sample, so the correct plan is one protocol cycle
followed by ``return_home``. The call goes through :meth:`System2Pipeline.reset` exactly as on the
robot (protocol check, retries, offline fallback), with the task prompt of the deployment config.

    python -m high_level_model.eval.eval_planner_recorded --repo_id <dataset> \\
        --skill_library configs/skill_library/bloodgas_recorded.yaml \\
        --prompts configs/skill_library/prompts/bloodgas.yaml --episodes_file <heldout.json> \\
        --llm_model qwen3.6-plus --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--skill_library", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--episodes_file", required=True, help='JSON with a "full_workflows" list')
    ap.add_argument("--camera", default="observation.images.cam_top")
    ap.add_argument("--task", default="blood gas sample analysis")
    ap.add_argument("--llm_model", default="qwen3.6-plus")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from high_level_model.data.high_level_dataset import _get_ep_bounds
    from high_level_model.planning.llm_backends import qwen_llm_fn
    from high_level_model.planning.planner import Planner
    from high_level_model.planning.prompt_templates import load_prompt_set
    from high_level_model.planning.skill_library import SkillLibrary
    from high_level_model.planning.system2_pipeline import System2Pipeline

    dset = LeRobotDataset(repo_id=args.repo_id, video_backend="pyav", tolerance_s=1e-2)
    with open(args.episodes_file) as f:
        episodes = json.load(f)["full_workflows"]
    library = SkillLibrary.from_file(args.skill_library)
    backend = qwen_llm_fn(model=args.llm_model, enable_thinking=False)
    expected = None
    rows = []
    for ep in episodes:
        start, _ = _get_ep_bounds(dset.meta, ep)
        img = (dset[start][args.camera].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        calls = []

        def llm_fn(prompt, images=None):
            t0 = time.time()
            reply = backend(prompt, images=images)
            calls.append({"latency_s": round(time.time() - t0, 3), "usage": backend.last_usage, "reply": reply[:400]})
            return reply

        planner = Planner(library, llm_fn=llm_fn, prompt_set=load_prompt_set(args.prompts))
        expected = expected or planner.cycle_ids() + planner.terminal_ids()
        pipe = System2Pipeline(library, planner, clock=lambda: 0.0)
        plan = pipe.reset(args.task, images=[img])
        rows.append({"episode": ep, "plan": plan, "correct": plan == expected, "admissible": pipe.plan_admissible,
                     "plan_source": pipe.plan_source, "calls": calls})
        print(json.dumps({k: rows[-1][k] for k in ("episode", "correct", "plan_source")}), flush=True)
    calls = [c for r in rows for c in r["calls"]]
    summary = {"llm_model": args.llm_model, "episodes": len(rows), "correct": sum(r["correct"] for r in rows),
               "admissible": sum(r["admissible"] for r in rows), "calls": len(calls),
               "latency_s_mean": float(np.mean([c["latency_s"] for c in calls])) if calls else None,
               "tokens_per_call_mean": float(np.mean([c["usage"]["total_tokens"] for c in calls if c.get("usage")]))
               if calls else None, "expected_plan": expected}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "planner_recorded.json"), "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
