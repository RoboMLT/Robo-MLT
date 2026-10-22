"""Train System 1 (Generative Executor) or any of the learning-based baselines.

The policy trained is selected by ``policy.type`` in the YAML config (see
``configs/system1/train/{qwen3vl_vla,pi0,pi05,act,smolvla}.yaml`` — the first is
Robo-MLT's System 1, the rest are the paper's baselines). All hyper-parameters come
from that YAML file; individual fields can be overridden on the command line with
``key=value`` syntax (dot-notation for nested keys, e.g. ``optimizer.lr=1e-4``).

Example::

    python -m low_level_model.training.train_system1 \\
        configs/system1/train/qwen3vl_vla.yaml

    # override fields
    python -m low_level_model.training.train_system1 \\
        configs/system1/train/qwen3vl_vla.yaml \\
        batch_size=16 dataset.repo_id=my/dataset optimizer.lr=1e-4
"""

# from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import format_big_number

from low_level_model.models.pi0.configuration_pi0 import PI0Config
from low_level_model.models.factory import make_policy, get_config_class, make_pre_post_processors as make_processors
from robomlt.config import load_yaml_with_overrides

logger = logging.getLogger(__name__)


# ───────────────────────────── Config dataclasses ────────────────────────────

@dataclass
class DatasetConfig:
    repo_id: str = ""
    root: Optional[str] = None
    video_backend: str = "pyav"
    use_imagenet_stats: bool = False
    # Exclude training samples whose action-chunk window overlaps a back_event (erroneous) frame, so
    # the VLA never imitates deliberately-bad recovery poses. No-op if the dataset has no back_event
    # column / no positives.
    filter_back_events: bool = True
    # Per-frame atomic-skill language conditioning. When `skill_library` (a
    # configs/skill_library/*.yaml path) is set, each frame's `subtask_index` is mapped to the
    # matching skill's `canonical_instruction` and injected as the `subtask` field, so the training
    # prompt becomes "task: <high> subtask: <atomic>" — matching what System 2 feeds at deployment.
    # `subtask_map` is an optional explicit {index: instruction} override (takes precedence over the
    # library). Leave both unset to keep the legacy single-prompt behaviour.
    skill_library: Optional[str] = None
    subtask_map: Optional[dict] = None
    subtask_index_column: str = "subtask_index"
    # Truncate each action chunk at its first subtask transition (via `subtask_index`), so steps
    # belonging to the *next* subtask are excluded from the loss instead of being supervised under
    # the current subtask's prompt. Prevents chunk-boundary action blur when chunk_size spans a
    # subtask boundary. Requires the `subtask_index` column; no-op if absent.
    mask_cross_subtask_actions: bool = False


@dataclass
class OptimizerConfig:
    lr: float = 2.5e-5
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    use_8bit: bool = False


@dataclass
class SchedulerConfig:
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6


@dataclass
class WandBConfig:
    enable: bool = False
    mode: str = "online"
    project: str = "RoboMLT_system1"
    entity: str = ""
    disable_artifact: bool = True


@dataclass
class System1TrainConfig:
    # Any registered System-1 policy config (PI0Config / ACTConfig / SmolVLAConfig /
    # Qwen3VLVLAConfig); `_to_config` builds the right one from `policy.type`.
    policy: PreTrainedConfig = field(default_factory=PI0Config)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output_dir: str = "outputs/system1"
    job_name: str = "pi0_system1"
    batch_size: int = 8
    steps: int = 50_000
    num_workers: int = 4
    seed: Optional[int] = None
    grad_accum_steps: int = 1
    resume: bool = False
    training_mode: str = "plain"
    # Temporal-delay augmentation: random action-chunk delay in [0, max_delay_steps].
    # 0 disables it. In "shared_observation" mode every valid offset is trained per
    # observation. In "ttrtc" mode it is the upper bound on the random prefix length
    # d ~ U{0..min(max_delay_steps, chunk_size)}.
    max_delay_steps: int = 0
    # Deprecated: prefer `training_mode`. Kept for backward compatibility — when True
    # and `training_mode` is left at its default, it maps to "shared_observation".
    shared_observation: bool = False
    save_checkpoint: bool = True
    save_freq: int = 5_000
    log_freq: int = 50
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


# ────────────────────────────── Config loading ───────────────────────────────

def _to_config(raw: dict) -> System1TrainConfig:
    """Convert a raw dict (from YAML) into a System1TrainConfig."""

    def _dc(cls, d):
        """Recursively fill a dataclass from a dict."""
        import dataclasses
        if d is None:
            return cls()
        if not dataclasses.is_dataclass(cls):
            return d
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                val = d[f.name]
                if dataclasses.is_dataclass(f.type) or (
                    isinstance(f.type, type) and dataclasses.is_dataclass(f.type)
                ):
                    val = _dc(f.type, val)
                kwargs[f.name] = val
        return cls(**kwargs)

    # ── policy ──
    # Dispatch on policy.type so the right config class is built (defaults to
    # pi0_system1 for backward compatibility). The remaining keys must all be
    # valid fields of that config dataclass.
    policy_raw = raw.pop("policy", {}) or {}
    policy_type = policy_raw.get("type") or "pi0_system1"
    policy_kwargs = {
        k: v for k, v in policy_raw.items()
        if k not in ("type", "push_to_hub") and v is not None
    }
    pretrained_path = policy_raw.get("pretrained_path")
    config_cls = get_config_class(policy_type)
    policy_cfg = config_cls(**policy_kwargs)
    if pretrained_path:
        policy_cfg.pretrained_path = pretrained_path

    # Whether `training_mode` was explicitly provided (before we build defaults),
    # so the legacy `shared_observation` flag only takes effect when it was not.
    training_mode_given = "training_mode" in raw

    cfg = System1TrainConfig(
        policy=policy_cfg,
        dataset=DatasetConfig(**{k: v for k, v in (raw.pop("dataset", {}) or {}).items()}),
        optimizer=OptimizerConfig(**{k: v for k, v in (raw.pop("optimizer", {}) or {}).items()}),
        scheduler=SchedulerConfig(**{k: v for k, v in (raw.pop("scheduler", {}) or {}).items()}),
        wandb=WandBConfig(**{k: v for k, v in (raw.pop("wandb", {}) or {}).items()}),
        **{k: v for k, v in raw.items() if k in System1TrainConfig.__dataclass_fields__},
    )
    _normalize_training_mode(cfg, training_mode_given)
    return cfg


_VALID_TRAINING_MODES = ("plain", "shared_observation", "ttrtc")


def _normalize_training_mode(cfg: System1TrainConfig, training_mode_given: bool) -> None:
    """Resolve/validate `training_mode` and propagate TTRTC settings to the policy.

    - Back-compat: a legacy `shared_observation: true` (with no explicit
      `training_mode`) maps to `training_mode="shared_observation"`.
    - Enforces mutual exclusivity between TTRTC and VLASH shared-observation.
    - When TTRTC is selected, records it on the policy config so the saved
      checkpoint (and inference) auto-detect it.
    """
    mode = cfg.training_mode

    # Legacy shim: only honor the old boolean when training_mode was not given.
    if not training_mode_given and cfg.shared_observation and mode == "plain":
        logger.warning(
            "`shared_observation: true` is deprecated; use `training_mode: shared_observation`. "
            "Mapping it for you."
        )
        mode = "shared_observation"

    if mode not in _VALID_TRAINING_MODES:
        raise ValueError(
            f"Invalid training_mode {mode!r}; expected one of {_VALID_TRAINING_MODES}."
        )

    # Guard against stale old-field + new-field conflicts (the enum itself is exclusive).
    if mode == "ttrtc" and cfg.shared_observation and training_mode_given:
        raise ValueError(
            "training_mode='ttrtc' is incompatible with shared_observation=true "
            "(TTRTC and VLASH shared-observation cannot be combined). Set "
            "shared_observation=false."
        )

    # Keep the boolean consistent with the resolved mode for the rest of the pipeline.
    cfg.shared_observation = mode == "shared_observation"
    cfg.training_mode = mode

    if mode == "ttrtc":
        policy_type = getattr(cfg.policy, "type", None)
        if policy_type not in ("pi0", "pi0_system1"):
            raise ValueError(
                f"training_mode='ttrtc' is only supported for the pi0 policy, got "
                f"policy.type={policy_type!r}."
            )
        # Record TTRTC on the model config so the checkpoint + inference auto-detect it.
        cfg.policy.ttrtc = True
        if cfg.max_delay_steps > 0:
            cfg.policy.ttrtc_max_delay = cfg.max_delay_steps


def load_config(yaml_path: str, overrides: list[str]) -> System1TrainConfig:
    """Load YAML config and apply ``key=value`` CLI overrides."""
    return _to_config(load_yaml_with_overrides(yaml_path, overrides))


# ───────────────────────────── Training helpers ──────────────────────────────

def resolve_mixed_precision(dtype: str | None) -> str:
    """Map the policy ``dtype`` from YAML to an Accelerate ``mixed_precision`` mode.

    The model holds its weights in ``dtype`` (bfloat16 transformer, float32 vision
    tower / norms via ``to_bfloat16_for_selected_params``).  We enable the matching
    autocast so the *entire* forward — including the flow-matching MSE loss — runs
    under AMP.  Autocast keeps reduction ops such as ``mse_loss`` in float32 with
    autograd-aware casts; this is what prevents the
    ``Found dtype Float but expected BFloat16`` backward error, with no manual
    casting anywhere in the model.

    Running via plain ``python -m`` (instead of ``accelerate launch``) means the
    Accelerator would otherwise default to ``"no"`` and disable autocast — which is
    exactly the bug this resolves: dtype now drives both the weight precision and
    the autocast mode from a single YAML field.
    """
    mapping = {
        "bfloat16": "bf16", "bf16": "bf16",
        "float16": "fp16", "fp16": "fp16", "half": "fp16",
        "float32": "no", "fp32": "no", "float": "no",
    }
    if dtype is None:
        return "no"
    return mapping.get(str(dtype).lower(), "no")


def _make_dataset(
    cfg: System1TrainConfig,
) -> tuple[LeRobotDataset, torch.utils.data.DataLoader, bool]:
    """Build the training dataset + dataloader.

    Selects between three modes from the YAML config:
      - max_delay_steps == 0                       → plain LeRobotDataset.
      - max_delay_steps > 0, shared_observation off → DelayAugmentedDataset
        (one random offset per sample).
      - max_delay_steps > 0, shared_observation on  → SharedObservationDataset
        (all offsets per observation, shared prefix) + custom collate.

    Returns ``(dataset, loader, use_shared_observation)``.
    """
    from low_level_model.data import (
        DelayAugmentedDataset,
        SharedObservationDataset,
        SubtaskInjectingDataset,
        build_back_filtered_indices,
        build_subtask_map,
        shared_observation_collate_fn,
    )

    meta = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root).meta
    fps = meta.fps
    delta_timestamps = {"action": [i / fps for i in range(cfg.policy.chunk_size)]}

    is_ttrtc = cfg.training_mode == "ttrtc"
    # TTRTC samples its action-prefix delay *inside* the model on the true (unshifted)
    # action chunk, so it must NOT use the delay/shared-observation datasets — those
    # shift the chunk and swap in the previous action as state. Force the plain path.
    use_delay = cfg.max_delay_steps > 0 and not is_ttrtc
    use_shared = use_delay and cfg.shared_observation
    mask_cross = cfg.dataset.mask_cross_subtask_actions
    collate_fn = None

    common = dict(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.dataset.video_backend,
    )
    delay_common = dict(
        mask_cross_subtask_actions=mask_cross,
        subtask_index_column=cfg.dataset.subtask_index_column,
        **common,
    )

    if use_shared:
        logger.info(
            "Dataset mode: shared-observation, max_delay_steps=%d "
            "(training all offsets [0, %d] per observation)",
            cfg.max_delay_steps, cfg.max_delay_steps,
        )
        dataset = SharedObservationDataset(max_delay_steps=cfg.max_delay_steps, **delay_common)
        collate_fn = shared_observation_collate_fn
    elif use_delay:
        logger.info(
            "Dataset mode: delay-augmented, max_delay_steps=%d (one random offset per sample)",
            cfg.max_delay_steps,
        )
        dataset = DelayAugmentedDataset(max_delay_steps=cfg.max_delay_steps, **delay_common)
    elif mask_cross:
        # No temporal delay, but cross-subtask masking still needs the DelayAugmentedDataset
        # query-index path (offset is always 0).
        logger.info("Dataset mode: plain + cross-subtask action masking (max_delay_steps=0)")
        dataset = DelayAugmentedDataset(max_delay_steps=0, **delay_common)
    elif is_ttrtc:
        logger.info(
            "Dataset mode: plain (TTRTC — prefix delay d~U{0..%d} sampled inside the model)",
            min(int(getattr(cfg.policy, "ttrtc_max_delay", cfg.max_delay_steps)), cfg.policy.chunk_size),
        )
        dataset = LeRobotDataset(**common)
    else:
        logger.info("Dataset mode: plain (no temporal delay)")
        dataset = LeRobotDataset(**common)

    # Optional per-frame atomic-skill language conditioning: map subtask_index → instruction and
    # inject it as the `subtask` field so train prompts match what System 2 feeds at deployment.
    if cfg.dataset.skill_library or cfg.dataset.subtask_map:
        if cfg.dataset.subtask_map:
            subtask_map = {int(k): str(v) for k, v in cfg.dataset.subtask_map.items()}
            src = "explicit subtask_map"
        else:
            subtask_map = build_subtask_map(cfg.dataset.skill_library)
            src = cfg.dataset.skill_library
        logger.info(
            "Subtask language injection enabled: %d skills from %s → %s",
            len(subtask_map), src, {i: subtask_map[i] for i in sorted(subtask_map)},
        )
        dataset = SubtaskInjectingDataset(
            dataset, subtask_map, column=cfg.dataset.subtask_index_column,
        )

    if cfg.dataset.use_imagenet_stats:
        from lerobot.datasets.factory import IMAGENET_STATS
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    # Optionally exclude samples whose action-chunk window overlaps a back_event (erroneous) frame.
    # When there is nothing to filter, fall back to plain shuffling over all frames.
    sampler = None
    allowed_indices = None
    if cfg.dataset.filter_back_events:
        # TTRTC does not shift the observation, so its action window is just chunk_size.
        back_filter_delay = 0 if is_ttrtc else cfg.max_delay_steps
        allowed_indices = build_back_filtered_indices(
            dataset, chunk_size=cfg.policy.chunk_size, max_delay_steps=back_filter_delay,
        )
    if allowed_indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(allowed_indices)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    return dataset, loader, use_shared


def _make_optimizer_and_scheduler(
    cfg: System1TrainConfig,
    policy: PreTrainedPolicy,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    ocfg = cfg.optimizer
    scfg = cfg.scheduler

    optim_params = policy.get_optim_params() if hasattr(policy, "get_optim_params") else policy.parameters()
    if getattr(ocfg, "use_8bit", False):
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "optimizer.use_8bit=true requires bitsandbytes. Install it with "
                "`pip install bitsandbytes`, or set optimizer.use_8bit=false."
            ) from e
        optimizer = bnb.optim.AdamW8bit(
            optim_params,
            lr=ocfg.lr,
            betas=tuple(ocfg.betas),
            eps=ocfg.eps,
            weight_decay=ocfg.weight_decay,
        )
        logger.info("Using bitsandbytes 8-bit AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=ocfg.lr,
            betas=tuple(ocfg.betas),
            eps=ocfg.eps,
            weight_decay=ocfg.weight_decay,
        )

    def lr_lambda(step: int) -> float:
        if step < scfg.warmup_steps:
            return step / max(1, scfg.warmup_steps)
        progress = (step - scfg.warmup_steps) / max(1, scfg.decay_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
        return (scfg.decay_lr + (ocfg.lr - scfg.decay_lr) * cosine_decay) / ocfg.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def auto_resume(cfg: System1TrainConfig) -> None:
    """Enable auto-resume if a previous checkpoint exists in output_dir."""
    if cfg.resume:
        return
    checkpoints_dir = Path(cfg.output_dir) / "checkpoints"
    last = checkpoints_dir / "last"
    if last.exists():
        cfg.resume = True
        logger.info("Auto-resume enabled from %s", last)
    elif Path(cfg.output_dir).is_dir() and not checkpoints_dir.is_dir():
        shutil.rmtree(cfg.output_dir, ignore_errors=True)
        logger.info("Removed stale output dir %s for a fresh run", cfg.output_dir)


def _save_checkpoint(cfg: System1TrainConfig, step: int, policy, optimizer, scheduler,
                     preprocessor, postprocessor) -> None:
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints" / f"{step:09d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dir = checkpoint_dir / "pretrained_model"
    pretrained_dir.mkdir(exist_ok=True)

    policy.save_pretrained(pretrained_dir)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / "training_state.pth",
    )
    try:
        preprocessor.save_pretrained(pretrained_dir)
        postprocessor.save_pretrained(pretrained_dir)
    except Exception:
        pass

    # Update "last" symlink.
    last = Path(cfg.output_dir) / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(checkpoint_dir.resolve())
    logger.info("Checkpoint saved at step %d → %s", step, checkpoint_dir)


def _load_training_state(cfg: System1TrainConfig, optimizer, scheduler) -> int:
    last = Path(cfg.output_dir) / "checkpoints" / "last"
    state_path = last / "training_state.pth"
    if not state_path.exists():
        logger.warning("No training state found at %s; starting from step 0", state_path)
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = state["step"]
    logger.info("Resumed from step %d", step)
    return step


def update_policy(
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    scheduler=None,
    *,
    loss_scale: float = 1.0,
    do_step: bool = True,
    training_mode: str = "plain",
) -> tuple[float, float | None, dict[str, float]]:
    """Single forward + backward; return total loss, grad norm, and loss components."""
    policy.train()
    with accelerator.autocast():
        if training_mode == "shared_observation":
            unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            loss, loss_details = unwrapped.forward_shared_observation(batch)
        elif training_mode == "ttrtc":
            unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            loss, loss_details = unwrapped.forward_ttrtc(batch)
        else:
            loss, loss_details = policy.forward(batch)
        raw_loss = loss.detach().item()
        scalar_loss_details: dict[str, float] = {}
        if isinstance(loss_details, dict):
            for name, value in loss_details.items():
                if isinstance(value, torch.Tensor):
                    scalar_loss_details[name] = value.detach().float().mean().item()
                elif isinstance(value, (int, float)):
                    scalar_loss_details[name] = float(value)
        loss = loss * loss_scale

    accelerator.backward(loss)

    grad_norm = None
    if do_step:
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm).item()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return raw_loss, grad_norm, scalar_loss_details


# ─────────────────────────────── Main train ──────────────────────────────────

def train(cfg: System1TrainConfig, accelerator: Accelerator | None = None) -> None:
    auto_resume(cfg)

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=resolve_mixed_precision(getattr(cfg.policy, "dtype", None)),
            kwargs_handlers=[ddp_kwargs],
        )

    # `force=True` is essential: importing lerobot/transformers installs a root
    # StreamHandler at WARNING level, which would make a plain basicConfig() a no-op
    # and silently suppress every logger.info(...) (the dataset/param summaries below).
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
        force=True,
    )
    is_main = accelerator.is_main_process

    # Performance: auto-tune cudnn kernels and allow TF32 matmuls on Ampere+.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main:
        logger.info(
            "Mixed precision: %s (from policy.dtype=%s)",
            accelerator.mixed_precision,
            getattr(cfg.policy, "dtype", None),
        )

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed + accelerator.process_index)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.wandb.enable and is_main:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.job_name,
            mode=cfg.wandb.mode,
            config=vars(cfg),
        )

    # ── Dataset ──────────────────────────────────────────────────────────────
    if is_main:
        logger.info("Creating dataset %s", cfg.dataset.repo_id)
    accelerator.wait_for_everyone()

    dataset, dataloader, use_shared_observation = _make_dataset(cfg)
    if is_main:
        n_episodes, n_frames = dataset.num_episodes, dataset.num_frames
        logger.info(
            colored("Dataset:", "cyan", attrs=["bold"])
            + " %s | %s episodes (%d) | %s steps/frames (%d) | fps=%d | ~%.0f steps/episode",
            cfg.dataset.repo_id,
            format_big_number(n_episodes), n_episodes,
            format_big_number(n_frames), n_frames,
            dataset.meta.fps,
            n_frames / max(1, n_episodes),
        )

    # ── Policy ───────────────────────────────────────────────────────────────
    if is_main:
        logger.info("Creating policy")

    policy_cfg = cfg.policy
    policy_cfg.device = str(accelerator.device)

    if getattr(policy_cfg, "pretrained_path", None):
        policy = make_policy(policy_cfg, dataset.meta)
    else:
        policy = make_policy(policy_cfg, dataset.meta)

    preprocessor, postprocessor = make_processors(
        policy_cfg, dataset_stats=dataset.meta.stats
    )

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    total_steps = cfg.steps
    optimizer, scheduler = _make_optimizer_and_scheduler(cfg, policy, total_steps)

    # ── Distributed preparation ───────────────────────────────────────────────
    policy, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, scheduler
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    step = 0
    if cfg.resume:
        step = _load_training_state(cfg, optimizer, scheduler)

    # ── Logging summary ──────────────────────────────────────────────────────
    if is_main:
        n_params = sum(p.numel() for p in policy.parameters())
        n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        n_frozen = n_params - n_train
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logger.info(
            colored("Policy params:", "cyan", attrs=["bold"])
            + " total=%s (%d) | trainable=%s (%d, %.1f%%) | frozen=%s (%d)",
            format_big_number(n_params), n_params,
            format_big_number(n_train), n_train, 100.0 * n_train / max(1, n_params),
            format_big_number(n_frozen), n_frozen,
        )
        logger.info(
            "Effective batch size: %d × %d × %d = %d",
            cfg.batch_size,
            accelerator.num_processes,
            cfg.grad_accum_steps,
            cfg.batch_size * accelerator.num_processes * cfg.grad_accum_steps,
        )
        logger.info("Training %d → %d steps", step, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    dl_iter = iter(dataloader)

    loss_window: list[float] = []
    grad_norm_window: list[float] = []
    component_loss_windows: dict[str, list[float]] = {
        "flow_mse_loss": [],
        "action_ce_loss": [],
        "subtask_ce_loss": [],
    }

    policy.train()
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps,
                disable=not is_main, dynamic_ncols=True)

    for _ in pbar:
        step_compute = 0.0
        micro_losses: list[float] = []
        micro_component_losses: dict[str, list[float]] = {
            name: [] for name in component_loss_windows
        }

        for micro_step in range(cfg.grad_accum_steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dataloader)
                batch = next(dl_iter)

            # The preprocessor only keeps observation.* + standard keys, so the
            # shared-observation extras would be dropped — preserve them around it.
            extra_keys = {
                k: batch[k] for k in ("offset_mask", "max_offsets") if k in batch
            }
            batch = preprocessor(batch)
            batch.update(extra_keys)
            if "offset_mask" in batch:
                batch["offset_mask"] = batch["offset_mask"].to(accelerator.device)

            do_step = micro_step == cfg.grad_accum_steps - 1

            t0 = time.perf_counter()
            raw_loss, grad_norm, loss_details = update_policy(
                policy, batch, optimizer, cfg.optimizer.grad_clip_norm,
                accelerator, scheduler=scheduler if do_step else None,
                loss_scale=1.0 / cfg.grad_accum_steps,
                do_step=do_step,
                training_mode=cfg.training_mode,
            )
            step_compute += time.perf_counter() - t0
            micro_losses.append(raw_loss)
            for name in micro_component_losses:
                if name in loss_details:
                    micro_component_losses[name].append(loss_details[name])

        step += 1
        loss_window.append(sum(micro_losses) / max(1, len(micro_losses)))
        for name, values in micro_component_losses.items():
            if values:
                component_loss_windows[name].append(sum(values) / len(values))
        if grad_norm is not None:
            grad_norm_window.append(grad_norm)

        # ── Logging ──────────────────────────────────────────────────────────
        if cfg.log_freq > 0 and step % cfg.log_freq == 0:
            avg_loss = sum(loss_window) / max(1, len(loss_window))
            avg_grad = sum(grad_norm_window) / max(1, len(grad_norm_window)) if grad_norm_window else 0.0
            lr = optimizer.param_groups[0]["lr"]
            if is_main:
                component_wandb_names = {
                    "flow_mse_loss": "loss/flow",
                    "action_ce_loss": "loss/fast",
                    "subtask_ce_loss": "loss/subtask",
                }
                component_metrics = {
                    wandb_name: sum(component_loss_windows[detail_name])
                    / len(component_loss_windows[detail_name])
                    for detail_name, wandb_name in component_wandb_names.items()
                    if component_loss_windows[detail_name]
                }
                is_pi05 = getattr(cfg.policy, "type", None) in ("pi05", "pi05_full")
                postfix_metrics = {
                    "loss": f"{avg_loss:.4f}",
                    "grad": f"{avg_grad:.3f}",
                    "lr": f"{lr:.2e}",
                }
                if is_pi05:
                    postfix_metrics.update(
                        flow=f"{component_metrics.get('loss/flow', float('nan')):.4f}",
                        fast=f"{component_metrics.get('loss/fast', float('nan')):.4f}",
                        subtask=f"{component_metrics.get('loss/subtask', float('nan')):.4f}",
                    )
                pbar.set_postfix(**postfix_metrics)
                if is_pi05:
                    missing_metrics = set(component_wandb_names.values()) - component_metrics.keys()
                    if missing_metrics:
                        logger.warning(
                            "PI0.5 did not return expected component losses at step %d: missing=%s",
                            step,
                            sorted(missing_metrics),
                        )
                    else:
                        logger.info(
                            "Step %d losses: total=%.6f flow=%.6f fast=%.6f subtask=%.6f",
                            step,
                            avg_loss,
                            component_metrics["loss/flow"],
                            component_metrics["loss/fast"],
                            component_metrics["loss/subtask"],
                        )
                if wandb_run:
                    wandb_metrics = {
                        "loss": avg_loss,
                        "grad_norm": avg_grad,
                        "lr": lr,
                        **component_metrics,
                    }
                    wandb_run.log(wandb_metrics, step=step)
            loss_window.clear()
            grad_norm_window.clear()
            for values in component_loss_windows.values():
                values.clear()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if is_main and cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == total_steps):
            _save_checkpoint(cfg, step, accelerator.unwrap_model(policy),
                             optimizer, scheduler, preprocessor, postprocessor)
            if wandb_run and not cfg.wandb.disable_artifact:
                last_dir = str(Path(cfg.output_dir) / "checkpoints" / "last")
                wandb_run.log_artifact(last_dir, type="model")

        accelerator.wait_for_everyone()

    if is_main:
        logger.info("Training complete. Checkpoints in %s", cfg.output_dir)
        if wandb_run:
            wandb_run.finish()

    accelerator.end_training()


# ─────────────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print(
            "Usage: python -m low_level_model.training.train_system1 "
            "<config.yaml> [key=value ...]\n"
            "Example: ... qwen3vl_vla.yaml batch_size=16 optimizer.lr=1e-4"
        )
        sys.exit(1)

    yaml_path = sys.argv[1]
    overrides = sys.argv[2:]
    cfg = load_config(yaml_path, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
