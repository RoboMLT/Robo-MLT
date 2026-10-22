# Robo-MLT: A Dual-System Robotic Medical Laboratory Technologist

Robo-MLT is a protocol-grounded, dual-system framework for long-horizon, contact-rich medical
laboratory manipulation. It separates explicit, low-frequency **workflow monitoring** (System 2)
from continuous, high-frequency **action generation** (System 1), so that visually similar
observations from different stages of a procedure (e.g. inserting vs. withdrawing a sample tube)
are disambiguated by *where the robot is in the protocol*, not by appearance alone.

This repository contains the implementation behind the accompanying paper. It is deliberately
scoped to the method described there: two medical-lab tasks (blood gas analysis, blood sample
tube sorting), five System-1 policies (the Qwen3-VL executor and four baselines), and the System-2
planner + completion gate + pointer controller with protocol constraints. The data-collection
workspace (teleoperation, recording, recovery-episode labelling) lives on the
[`data-collection`](https://github.com/RoboMLT/Robo-MLT/tree/data-collection) branch.


## Repository Layout

```
configs/                    All YAML configs
  skill_library/            atomic-skill libraries + protocol constraints, planner prompts
  system1/train/<task>/     qwen3vl_vla (System 1), pi0, pi05, act, smolvla (baselines)
  system2/train/            completion gate, one config per task (+ *_single_frame ablation)
  inference/                real-robot deployment, one config per task (+ pi05.yaml baseline)
  accelerate/               multi-GPU launch configs
high_level_model/           System 2 — planner, completion gate, pointer controller
  planning/                 SkillLibrary, protocol constraints, Planner, System2Pipeline, prompt templates, LLM backend
  models/                   siglip_encoder.py (f_phi/h_theta), completion_gate.py (g_psi, gamma_t, beta_t)
  data/                     CompletionGateDataset, release builder, dataset statistics, merge tools
  training/                 train_completion_gate.py
  eval/                     gate_metrics.py, eval_completion_gate_video.py, figure scripts
  tests/                    test_system2.py, test_dataset_tools.py
low_level_model/            System 1 — policies, async runtime, real-robot deployment
  models/                   qwen3vl_vla/ (System 1), pi0/, pi05/, act/, smolvla/ (baselines), factory.py
  runtime/                  inference_system1.py, action_smoothing.py
  robot/                    robot_inference.py (deployment entry point), system2_controller.py
  training/                 train_system1.py
  tests/                    test_system1.py, test_pi05.py
tests/                      config consistency checks
```

## Installation

```bash
git clone https://github.com/RoboMLT/Robo-MLT.git
cd Robo-MLT
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU for training/inference. See `requirements.txt` for
version notes (in particular: Qwen3-VL needs `transformers>=4.57`; with an older version, point
`policy.vlm_model_id` at a Qwen2.5-VL checkpoint instead).

**Real-robot deployment and dataset loading** (`low_level_model/robot/robot_inference.py`, and
anything that goes through `lerobot.datasets`) additionally need a fork of
[LeRobot](https://github.com/huggingface/lerobot) with `piper` / `dual_piper` robot support
(motor bus + calibration) that is not in the upstream package. System-2 planning/gate logic
(`high_level_model/planning/`, `models/completion_gate.py`, `robot/system2_controller.py`) has no
LeRobot dependency and runs standalone — `python -m high_level_model.tests.test_system2` needs
nothing beyond `requirements.txt`.

**LLM planner:** set `DASHSCOPE_API_KEY` (Alibaba Cloud DashScope, Qwen3.7-Max) to enable the real
planner and replanner. Without it, `Planner` falls back to one fixed protocol cycle per remaining
work item, which keeps every offline test and the control-logic unit tests runnable without any
API key.

## Data Format

Datasets are [LeRobot v3](https://github.com/huggingface/lerobot) datasets with two Robo-MLT
extensions read decode-free from `hf_dataset` columns:

- `subtask_index` (+ `meta/subtasks.parquet` mapping index → canonical instruction text) — the
  atomic skill active at each frame. The text must match the skill library byte for byte.
- `back_event` — 1 on the final $K_{rec} \sim U[20, 40]$ frames of a recovery episode, 0
  elsewhere; supervises the recovery head $\beta_t$ (stored as `back_head`). It is labelled
  offline by `label_recovery_tail.py` on the `data-collection` branch.

Each task's data has two corpora. The nominal corpus holds complete long-horizon demonstrations
(start/end of every atomic skill annotated online) and single-atomic-skill episodes from
randomized initial poses (broadens per-skill state coverage); it trains System 1, the baselines and
System 2. The recovery corpus holds episodes ended by an operator intervention or a teleoperation
failure (e.g. a dropped tube) and is used by System 2 only. `action` is the follower arm's measured
joint position (6 joints + gripper per arm).

`high_level_model/data/build_release_dataset.py` merges multiple capture sessions into one release
dataset with de-duplication and schema reconciliation; `high_level_model/data/dataset_stats.py`
reports the distinct episode counts (complete workflows, atomic-skill episodes, recovery episodes)
and checks the subtask text against the skill library.

## Training

```bash
# System 1 — Robo-MLT's executor (Qwen3-VL + flow matching)
python -m low_level_model.training.train_system1 configs/system1/train/blood_gas_analysis/qwen3vl_vla.yaml

# System 1 baselines
python -m low_level_model.training.train_system1 configs/system1/train/blood_gas_analysis/pi0.yaml
python -m low_level_model.training.train_system1 configs/system1/train/blood_gas_analysis/pi05.yaml
python -m low_level_model.training.train_system1 configs/system1/train/blood_gas_analysis/act.yaml
python -m low_level_model.training.train_system1 configs/system1/train/blood_gas_analysis/smolvla.yaml

# System 2 — completion gate (nominal + recovery corpus)
python -m high_level_model.training.train_completion_gate \
    configs/system2/train/blood_gas_analysis.yaml \
    dataset.repo_id=<path/to/data>/blood_gas_analysis \
    dataset.recovery_repo_id=<path/to/data>/blood_gas_analysis_recovery
```

Replace `blood_gas_analysis` with `tube_sorting` for the tube-sorting task. All configs are
YAML-first with dot-notation CLI overrides, e.g.
`... blood_gas_analysis.yaml max_steps=30000 loss.lambda_rec=0.5`. Multi-GPU System-1 training runs
through `accelerate launch --config_file configs/accelerate/zero3.yaml`. The completion-gate run
folder holds `completion_gate_best.pth` and a `completion_gate_config.yaml` sidecar that deployment
reads (architecture, history length, camera order, trained skill texts).

## Evaluation

```bash
# System-2 unit tests (planner fallback, protocol constraints, gate shapes, pipeline and
# controller logic — the torch-dependent tests skip gracefully if torch is unavailable)
python -m high_level_model.tests.test_system2

# System-1 unit tests (async streamer, chunk blending, flow-matching shapes, RTC/TTRTC)
python -m low_level_model.tests.test_system1

# Config consistency and dataset tools
python -m tests.test_paper_alignment
python -m high_level_model.tests.test_dataset_tools

# Per-episode completion-gate visualisation (predicted vs. GT completion/recovery curves)
python -m high_level_model.eval.eval_completion_gate_video \
    --repo_id <path/to/data>/blood_gas_analysis \
    --gate_ckpt outputs/system2/blood_gas_analysis/<run>/completion_gate_best.pth --episode 0

# Closed-loop switch-timing metric (median advance offset vs. GT segment boundaries)
# — see high_level_model/eval/gate_metrics.py::closed_loop_switch_report
```

`high_level_model/eval/plot_inference_demo.py` and `plot_back_recovery_demo.py` reproduce the
closed-loop advance/recovery figures by replaying recorded episodes offline through the trained
gate; `plot_demo_rows.py` composes their cached results into the combined multi-task figure (with
an optional editable `.pptx` export).

## Deployment

```bash
# Real-robot inference: System 1 + System 2 (one config per task)
python -m low_level_model.robot.robot_inference configs/inference/blood_gas_analysis.yaml \
    policy_path=<System-1 checkpoint>/pretrained_model gate_ckpt=<System-2 run folder>

# Ablation variants: ours | single_frame | no_recovery | timeout_only
python -m low_level_model.robot.robot_inference configs/inference/blood_gas_analysis.yaml variant=no_recovery

# pi0.5 standalone baseline (no external System-2 gate — it decodes its own subtask)
python -m low_level_model.robot.robot_inference_pi05 configs/inference/pi05.yaml
```

`single_frame` loads a gate trained with `configs/system2/train/<task>_single_frame.yaml`. During
deployment, digit keys jump the plan pointer to a skill (operator override), `0` prints the plan
and `q` / `Esc` stops; the arm holds its position while the replanner runs.

## Limitations

System 1 and System 2 are trained in separate stages from the same demonstrations. The skill
library is extensible only in the sense that the frozen planner can re-order/repeat existing
skills; adding a genuinely new laboratory operation needs new System-1 demonstrations and further
System-2 training. The recovery head only covers the failure families present in its training
data. The two-task real-robot evaluation uses the same workspace/object distributions as data
collection.

## Citation

```bibtex
@article{robomlt2026,
  title   = {Robo-MLT: A Dual-System Robotic Medical
             Laboratory Technologist for Generative
             Long-Horizon Manipulation},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```

## License

Apache-2.0 — see `LICENSE`. Several modules port or adapt code from other open-source projects
(LeRobot, openpi, starVLA, kai0); see `THIRD_PARTY_NOTICES.md` for details and their upstream
license terms.
