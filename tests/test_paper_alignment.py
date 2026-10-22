"""The shipped configs must carry the values reported in the paper (Table I, Table II, Sec. III-A).

Run:  python -m tests.test_paper_alignment   (from the repo root; needs only PyYAML)

Besides the values, every YAML key is checked against the dataclass that consumes it (read from
the source with ``ast``, so neither torch nor the LeRobot fork is needed).
"""

import ast
import glob
import os
from functools import cache

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ("blood_gas_analysis", "tube_sorting")
METHODS = ("qwen3vl_vla", "pi0", "pi05", "act", "smolvla")


def _yaml(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def _fields(rel_path, class_name):
    """Annotated class attributes of ``class_name`` (and its in-file dataclass bases)."""
    with open(os.path.join(ROOT, rel_path), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    out, todo = set(), [class_name]
    while todo:
        node = classes[todo.pop()]
        out |= {s.target.id for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
        todo += [b.id for b in node.bases if isinstance(b, ast.Name) and b.id in classes]
    return frozenset(out)


# Fields every lerobot PreTrainedConfig accepts (base class lives outside the repo).
_PRETRAINED = {"device", "pretrained_path", "push_to_hub", "type", "n_obs_steps", "use_amp",
               "input_features", "output_features"}
_POLICY_CFG = {
    "qwen3vl_vla": ("low_level_model/models/qwen3vl_vla/configuration_qwen3vl_vla.py", "Qwen3VLVLAConfig"),
    "pi0": ("low_level_model/models/pi0/configuration_pi0.py", "PI0Config"),
    "pi05": ("low_level_model/models/pi05/configuration_pi05.py", "PI05FullConfig"),
}


def _check_keys(section, allowed, where):
    unknown = set(section) - set(allowed)
    assert not unknown, f"{where}: unknown keys {sorted(unknown)}"


def test_runtime_table_i() -> bool:
    """Table I: f_ctrl, N, r, w, K, M, thresholds/debounces, T_max, Delta_cd, B_replan."""
    tr = "low_level_model/robot/robot_inference.py"
    for task in TASKS:
        cfg = _yaml(f"configs/inference/{task}.yaml")
        _check_keys(cfg, _fields(tr, "InferenceConfig"), f"inference/{task}")
        _check_keys(cfg["pipeline"], _fields(tr, "PipelineConfig"), f"inference/{task}.pipeline")
        _check_keys(cfg["smoothing"], _fields("low_level_model/robot/runtime_builder.py",
                                              "SmoothingConfig"), f"inference/{task}.smoothing")
        assert cfg["fps"] == 20 and cfg["overlap_steps"] == 8
        assert cfg["smoothing"]["enabled"] and cfg["smoothing"]["min_smooth_steps"] == 4
        assert cfg["sampling_interval"] == 15 and cfg["cooldown_steps"] == 10
        p = cfg["pipeline"]
        assert (p["tau_done"], p["k_a"], p["tau_rec"], p["k_b"]) == (0.6, 2, 0.8, 3)
        assert p["t_max_s"] == 30.0 and p["replan_budget"] == 3
        assert (p["rec_ema_alpha"], p["completion_median_window"], p["completion_ema_alpha"]) \
            == (1.0, 1, 1.0), "the paper debounces raw probabilities"
        assert cfg["variant"] == "ours" and cfg["use_llm"] and cfg["llm_model"] == "qwen3.7-max"
        assert len(cfg["camera_names"]) == 3 and cfg["robot"]["type"] == "dual_piper"
        policy = _yaml(f"configs/system1/train/{task}/qwen3vl_vla.yaml")["policy"]
        assert policy["n_action_steps"] == 30, "N = 30"
    pi05 = _yaml("configs/inference/pi05.yaml")
    assert pi05["fps"] == 20 and len(pi05["camera_names"]) == 3, "baselines share rate and cameras"
    print("   Table I values OK for both tasks; baseline shares f_ctrl and camera streams")
    return True


def test_system2_table_ii() -> bool:
    tr = "high_level_model/training/train_completion_gate.py"
    for task in TASKS:
        for suffix, m in (("", 6), ("_single_frame", 1)):
            cfg = _yaml(f"configs/system2/train/{task}{suffix}.yaml")
            where = f"system2/{task}{suffix}"
            _check_keys(cfg, _fields(tr, "CompletionGateTrainConfig"), where)
            for sec, cls in (("dataset", "DatasetConfig"), ("sampler", "SamplerConfig"),
                             ("model", "ModelConfig"), ("optimizer", "OptimizerConfig"),
                             ("loss", "LossConfig"), ("cache", "CacheConfig"),
                             ("wandb", "WandBConfig")):
                _check_keys(cfg[sec], _fields(tr, cls), f"{where}.{sec}")
            assert cfg["dataset"]["history_len"] == m
            s = cfg["sampler"]
            assert s["segs_per_batch"] * s["frames_per_seg"] == 128 == cfg["batch_size"]
            assert cfg["max_steps"] == 60000
            o = cfg["optimizer"]
            assert o["lr"] == 1e-4 and o["schedule"] == "cosine"
            assert cfg["loss"]["lambda_rec"] == cfg["loss"]["lambda_completion"] == 1.0
            assert cfg["loss"]["rec_pos_weight"] is None and cfg["loss"]["completion_pos_weight"] is None
            mdl = cfg["model"]
            assert mdl["freeze_siglip"] and mdl["fusion"] == "cross_attention"
            assert mdl["temporal_layers"] == 2 and mdl["cross_attn_layers"] == 2
            assert cfg["dataset"]["recovery_repo_id"], "recovery corpus supervises beta_t"
    print("   Table II System-2 column OK (+ the M = 1 Single-Frame configs)")
    return True


def test_system1_table_ii() -> bool:
    tr = "low_level_model/training/train_system1.py"
    for task in TASKS:
        for method in METHODS:
            cfg = _yaml(f"configs/system1/train/{task}/{method}.yaml")
            where = f"system1/{task}/{method}"
            _check_keys(cfg, _fields(tr, "System1TrainConfig"), where)
            for sec, cls in (("dataset", "DatasetConfig"), ("optimizer", "OptimizerConfig"),
                             ("scheduler", "SchedulerConfig"), ("wandb", "WandBConfig")):
                _check_keys(cfg[sec], _fields(tr, cls), f"{where}.{sec}")
            if method in _POLICY_CFG:
                _check_keys(cfg["policy"], _fields(*_POLICY_CFG[method]) | _PRETRAINED,
                            f"{where}.policy")
            assert cfg["steps"] == 80000 and cfg["batch_size"] == 8, where
            assert cfg["policy"]["chunk_size"] == cfg["policy"]["n_action_steps"] == 30, where
            sch = cfg["scheduler"]
            assert sch["warmup_steps"] + sch["decay_steps"] == cfg["steps"], "cosine spans training"
            assert not cfg["wandb"]["entity"] and not cfg["wandb"]["enable"]
            assert cfg["dataset"]["repo_id"].endswith(f"RoboMLT-Lab/{task}")
            if method in ("qwen3vl_vla", "pi0", "smolvla"):
                assert cfg["dataset"]["skill_library"], f"{where}: atomic-skill instruction"
        q = _yaml(f"configs/system1/train/{task}/qwen3vl_vla.yaml")["policy"]
        assert q["vlm_model_id"] == "Qwen/Qwen3-VL-2B-Instruct" and not q["freeze_vlm"]
        assert q["optimizer_lr"] == q["vlm_lr"] == 1e-4
    print("   Table II System-1 column OK; every baseline shares steps / batch / chunk")
    return True


def test_skill_libraries() -> bool:
    for rel, n in (("configs/skill_library/bloodgas.yaml", 8),
                   ("configs/skill_library/bloodtube_sorting.yaml", 5)):
        lib = _yaml(rel)
        assert len(lib["skills"]) == n, f"{rel}: N_s = {len(lib['skills'])}"
        assert lib["protocol"]["rules"] and lib["protocol"]["terminal"] == ["return_home"]
    print("   N_s = 8 (blood gas) / 5 (tube sorting); protocol constraints declared")
    return True


def test_all_yaml_parse() -> bool:
    files = glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True)
    for f in files:
        with open(f, encoding="utf-8") as fh:
            yaml.safe_load(fh)
    print(f"   {len(files)} YAML files parse")
    return True


def main():
    tests = [("all YAML parse", test_all_yaml_parse),
             ("Table I runtime", test_runtime_table_i),
             ("Table II System 2", test_system2_table_ii),
             ("Table II System 1", test_system1_table_ii),
             ("skill libraries", test_skill_libraries)]
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            print(f"--- {name}: PASS")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"--- {name}: FAIL ({exc})")
    print(f"\n==== summary: {len(tests) - failed} passed, {failed} failed ====")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
