# Robo-MLT: A Dual-System Robotic Medical Laboratory Technologist

Robo-MLT is a protocol-grounded, dual-system framework for long-horizon, contact-rich medical
laboratory manipulation. It separates explicit, low-frequency **workflow monitoring** (System 2)
from continuous, high-frequency **action generation** (System 1), so that visually similar
observations from different stages of a procedure (e.g. inserting vs. withdrawing a sample tube)
are disambiguated by *where the robot is in the protocol*, not by appearance alone.

This repository contains the implementation behind the accompanying paper. It is deliberately
scoped to the method described there: two medical-lab tasks (blood gas analysis, blood sample
tube sorting), five System-1 VLA policies, and the System-2 planner + completion-gate + pointer
controller.


## Repository Layout

```
configs/                    All YAML configs (see below)
high_level_model/           System 2 — planner, completion gate, pointer controller
  planning/                 SkillLibrary, Planner, System2Pipeline, prompt templates, LLM backend
  models/                   siglip_encoder.py (f_phi/h_theta), completion_gate.py (g_psi, gamma_t, beta_t)
  data/                     HighLevelSequenceDataset, CompletionGateDataset, annotation/merge tools
  training/                 train_competion_gate.py
  eval/                     gate_metrics.py, eval_completion_gate_video.py, paper-figure scripts
  tests/                    test_system2.py
low_level_model/            System 1 — policies, async runtime, real-robot deployment
  models/                   qwen3vl_vla/ (System 1), pi0/, pi05/, act/, smolvla/ (baselines), factory.py
  runtime/                  inference_system1.py, action_smoothing.py
  robot/                    robot_inference.py (deployment entry point), collect_dagger_dataset.py
  training/                 train_system1.py
  tests/                    test_system1.py, test_pi05.py, test_dagger_recorder.py
```

## Installation

```bash
git clone https://github.com/SkyLineHXY/Robo-MLT.git
cd Robo-MLT
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU for training/inference. See `requirements.txt` for
version notes (in particular: Qwen3-VL needs `transformers>=4.57`; with an older version, point
`policy.vlm_model_id` at a Qwen2.5-VL checkpoint instead).

**Real-robot deployment only** (`low_level_model/robot/`, and any dataset loading through
`lerobot.datasets`) additionally needs a fork of
[LeRobot](https://github.com/huggingface/lerobot) with `piper` / `dual_piper` robot support
(motor bus + calibration) that is not in the upstream package. System-2 planning/gate logic
(`high_level_model/planning/`, `models/completion_gate.py`) has no LeRobot dependency and runs
standalone — `python -m high_level_model.tests.test_system2` needs nothing beyond
`requirements.txt`.

**LLM planner (optional):** set `DASHSCOPE_API_KEY` (Alibaba Cloud DashScope / Qwen) to enable the
real planner. Without it, `Planner` falls back to the skill library's declaration order, which
keeps every offline test and the control-logic unit tests runnable without any API key.

## Data Format

Datasets are [LeRobot v3](https://github.com/huggingface/lerobot) datasets with two Robo-MLT
extensions read decode-free from `hf_dataset` columns:

- `subtask_index` (+ `meta/subtasks.parquet` mapping index → canonical instruction text) — the
  atomic skill active at each frame.
- `back_event` — 1 on frames where a human correction/intervention makes the active skill
  no longer completable; supervises the back head $\beta_t$. `data/annotate_back_windows.py` is an
  offline OpenCV tool for adding this column post hoc when it wasn't recorded inline.

Each task's data has two parts: complete long-horizon demonstrations (start/end of every atomic
skill annotated online) and single-atomic-skill episodes from randomized initial poses (broadens
per-skill state coverage). `high_level_model/data/merge_lerobot_datasets.py` merges multiple
capture sessions into one release dataset with de-duplication and schema reconciliation.

## Training

```bash
# System 1 — Robo-MLT's executor (Qwen3-VL + flow matching)
python -m low_level_model.training.train_system1 configs/system1/train/qwen3vl_vla.yaml

# System 1 baselines
python -m low_level_model.training.train_system1 configs/system1/train/pi0.yaml
python -m low_level_model.training.train_system1 configs/system1/train/pi05.yaml
python -m low_level_model.training.train_system1 configs/system1/train/act.yaml
python -m low_level_model.training.train_system1 configs/system1/train/smolvla.yaml

# System 2 — completion gate (two tasks, one config each)
python -m high_level_model.training.train_competion_gate \
    configs/system2/train/completion_gate.yaml            # tube sorting
python -m high_level_model.training.train_competion_gate \
    configs/system2/train/completion_gate_bloodgas.yaml    # blood gas analysis
```

All configs are YAML-first with dot-notation CLI overrides, e.g.
`... completion_gate.yaml num_epochs=30 loss.lambda_back=0.5`.

## Evaluation

```bash
# System-2 unit tests (planner fallback, gate shapes, pipeline control logic — no GPU required
# beyond the two torch-dependent gate tests, which skip gracefully if torch is unavailable)
python -m high_level_model.tests.test_system2

# System-1 unit tests (async streamer, chunk blending, flow-matching shapes, RTC/TTRTC)
python -m low_level_model.tests.test_system1

# Per-episode completion-gate visualisation (predicted vs. GT completion/back curves)
python -m high_level_model.eval.eval_completion_gate_video \
    --repo_id <path/to/data>/TubeSort_20260718 \
    --gate_ckpt outputs/completion_gate/<run>/completion_gate_best.pth --episode 0

# Closed-loop switch-timing metric (median advance offset vs. GT segment boundaries)
# — see high_level_model/eval/gate_metrics.py::closed_loop_switch_report
```

`high_level_model/eval/plot_inference_demo.py` and `plot_back_recovery_demo.py` reproduce the
paper's closed-loop advance/recovery figures; `plot_demo_rows.py` composes their cached results
into the combined multi-task figure (with an optional editable `.pptx` export).

## Deployment

```bash
# Real-robot inference: System 1 + optional System 2 (set gate_ckpt/skill_library to enable)
python -m low_level_model.robot.robot_inference configs/system1/inference/inference.yaml

# pi0.5 standalone baseline (no external System-2 gate — it decodes its own subtask)
python -m low_level_model.robot.robot_inference_pi05 configs/system1/inference/inference_pi05.yaml

# DAgger data collection (teleop takeover + subtask labelling)
python -m low_level_model.robot.collect_dagger_dataset configs/system1/inference/collect_dagger.yaml
```

## Limitations

System 1 and System 2 are trained in separate stages from the same demonstrations. The skill
library is extensible only in the sense that the frozen planner can re-order/repeat existing
skills; adding a genuinely new laboratory operation needs new System-1 demonstrations and further
System-2 training. The back head only covers the failure families present in its training data.
The two-task real-robot evaluation uses the same workspace/object distributions as data
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
