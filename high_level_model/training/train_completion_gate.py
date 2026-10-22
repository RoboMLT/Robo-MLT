"""Train the System 2 execution monitor (paper Sec. III-B, Eq. 6, Table II).

The SigLIP2 image/text encoders stay frozen; the skill-grounding trunk g_psi (causal temporal
encoder + cross-attention blocks) and the two heads are trained jointly in a single stage:

    L_S2 = lambda_c * BCE_wc(gamma) + lambda_r * BCE_wr(beta)

with w_c, w_r the inverse positive-class frequency of each label on the training split.
Table II: AdamW, learning rate 1e-4 with cosine decay, batch size 128, 60k steps,
lambda_c = lambda_r = 1. Checkpoints are selected by the completion BCE (the advance signal).

The nominal corpus (``dataset.repo_id``) supplies completion labels; the separately recorded
intervention / failure corpus (``dataset.recovery_repo_id``) supplies the recovery positives.

Validation is optional: ``dataset.train_ratio >= 1.0`` trains on every episode, selection then
falls back to the train metrics, and a rolling ``completion_gate_last.pth`` is written regardless.

All hyper-parameters come from a YAML config; override fields with ``key=value`` (dot-notation
for nested keys, e.g. ``optimizer.lr=1e-4``).

Example::

    python -m high_level_model.training.train_completion_gate \\
        configs/system2/train/blood_gas_analysis.yaml \\
        dataset.repo_id=/path/to/RoboMLT-Lab/blood_gas_analysis \\
        dataset.recovery_repo_id=/path/to/RoboMLT-Lab/blood_gas_analysis_recovery
"""

import dataclasses
import hashlib
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm

from high_level_model.data.completion_gate_dataset import (
    CompletionGateDataset,
    ConcatGateDataset,
    SegmentBatchSampler,
    make_completion_gate_loader,
)
from high_level_model.data.high_level_dataset import _get_ep_bounds
from high_level_model.models.completion_gate import CompletionGate
from high_level_model.models.siglip_encoder import build_siglip_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ───────────────────────────── Config dataclasses ────────────────────────────

@dataclass
class DatasetConfig:
    repo_id: str = "<path/to>/RoboMLT-Lab/blood_gas_analysis"
    # Separately recorded intervention / failure corpus (paper Sec. IV-A-3): the only source of
    # recovery-head positives. Its episodes are split train/val with the same ratio and seed.
    recovery_repo_id: Optional[str] = None
    camera_names: List[str] = field(default_factory=lambda: ["observation.images.cam_top"])
    history_len: int = 1
    prediction_offset: int = 0
    history_skip_frame: int = 4
    # Train-time random inter-frame stride (anti-overfit): draw the history stride per sample from
    # U(1, Δ) where Δ = random_skip_range. 0 disables (fixed history_skip_frame). Val stays
    # deterministic (uses history_skip_frame). Only meaningful when history_len > 1.
    random_skip_range: int = 0
    train_ratio: float = 0.85
    max_episodes: Optional[int] = None
    # Keep only episodes whose length (in frames) is strictly greater than this. 0 disables. Applied
    # to the training pool AND (when val_episodes is null) to the cross-dataset val pool, before the
    # train/val split — so short partial captures are dropped from both. The disk feature cache keys
    # on the resulting episode list, so a different filter transparently produces a fresh cache dir.
    min_episode_len: int = 0
    # Also keep every episode that contains a back_event frame, regardless of length (UNION with
    # ``min_episode_len``). Only matters when recovery episodes live inside ``repo_id`` itself.
    include_back_episodes: bool = False
    # Max allowed gap (s) between a requested and the nearest available video frame before lerobot
    # raises FrameTimestampError. lerobot's default (1e-4) is tighter than the timestamp rounding
    # drift in some captures (~2e-4 s), which aborts precompute mid-run. 1e-2
    # is a quarter of the 25 fps frame interval — well above the drift, far below grabbing a wrong
    # neighbouring frame. None keeps the lerobot default.
    tolerance_s: Optional[float] = None
    # Validate on a DIFFERENT dataset (true held-out generalization, not just held-out episodes of
    # the training capture). When set, the whole of repo_id trains and the split below is bypassed.
    # ``val_episodes`` picks the episodes by index within val_repo_id; null = all of them.
    val_repo_id: Optional[str] = None
    val_episodes: Optional[List[int]] = None
    # RNG seed for the episode shuffle + sub-sample (change it to draw a different random subset;
    # fixed by default so a run is reproducible).
    episode_subset_seed: int = 42
    use_command_in_meta: bool = False
    # Optional: skill-library YAML to cross-check against meta/subtasks at startup (warns on drift
    # between the trained skill strings and the ones the deployed planner will emit).
    skill_library: Optional[str] = None
    # JSON: [{episode, frame_start, frame_end}, ...] (episode-local frames); back targets default to 0.
    # Used only as a fallback when the dataset has no inline back_event column.
    back_annotations: Optional[str] = None
    # Prefer the inline ``back_event`` column for recovery-head labels; falls back to the
    # back_annotations JSON above when the column is absent.
    back_from_column: bool = True
    # Drop contiguous subtask runs shorter than this many frames (label-boundary noise) and prune
    # their samples. 0/1 disables. Episodes may carry only a subset of the task's subtasks
    # (partial captures); each run is segmented independently.
    min_segment_len: int = 0
    # Last N frames of each *truly-completed* segment are completion-head positives. Trades off
    # advance latency vs positive sparsity; tune against typical segment length.
    completion_window: int = 5
    # Also treat the episode's FINAL segment as completed (its last frames become positives). Turn on
    # when captures run to completion rather than being cut off mid-skill — without it a
    # single-subtask episode contributes zero positives and the head learns "this skill never ends".
    completion_at_episode_end: bool = False


@dataclass
class SamplerConfig:
    # group each batch into a few segments × several frames so positives are balanced within a
    # batch for the weighted BCE losses (batch_size = segs_per_batch * frames_per_seg)
    use_segment_sampler: bool = True
    segs_per_batch: int = 4
    frames_per_seg: int = 4


@dataclass
class ModelConfig:
    # Freeze the SigLIP2 image/text towers (paper: "the SigLIP2 image and text encoders remain
    # frozen"). False fine-tunes the visual tower too.
    freeze_siglip: bool = True
    # Trunk g_psi (Fig. 3): "cross_attention" = causal temporal encoder followed by cross-attention
    # blocks (text query, temporal keys/values). "concat" / "legacy_cross_attention" only reload
    # pre-release checkpoints.
    fusion: str = "cross_attention"
    temporal_layers: int = 2
    temporal_heads: int = 8
    cross_attn_layers: int = 2
    cross_attn_heads: int = 8


@dataclass
class OptimizerConfig:
    # AdamW (Table II). ``schedule: cosine`` decays lr to ``min_lr`` over ``max_steps`` after a
    # linear warm-up of ``warmup_steps``; ``constant`` keeps lr fixed.
    lr: float = 1e-4
    weight_decay: float = 1e-4
    schedule: str = "cosine"
    warmup_steps: int = 0
    min_lr: float = 0.0


@dataclass
class LossConfig:
    # lambda_r / lambda_c (paper Eq. 6): weights on the recovery- and completion-head BCE.
    lambda_rec: float = 1.0
    lambda_completion: float = 1.0
    # w_r / w_c: positive-class weights. None -> inverse positive-class frequency on the training
    # split (see compute_pos_weight / compute_completion_pos_weight).
    rec_pos_weight: Optional[float] = None
    completion_pos_weight: Optional[float] = None


# Pre-release config keys, still accepted: {section: {old: new}}.
_LEGACY_KEYS = {
    "loss": {"lambda_back": "lambda_rec", "back_pos_weight": "rec_pos_weight"},
}


@dataclass
class CacheConfig:
    # Precompute the frozen-backbone visual features once, then train heads (+ temporal encoder)
    # on the cached tensors — skips video decode + SigLIP/trunk forward every epoch. Auto-disabled
    # (with a warning) unless the backbone is fully frozen, since otherwise the features change.
    features: bool = True
    # Memoise the frozen SigLIP skill-text embeddings (handful of unique strings). Auto-disabled
    # inside the model when the SigLIP tower is being fine-tuned.
    text_embeddings: bool = True
    # Batch size used only for the one-off precompute sweep (independent of training batch_size).
    precompute_batch_size: int = 64
    # Where the frame-level caches are persisted. Keyed by a hash of everything that affects the
    # features, so parallel sweep processes hit the same directory and never recompute.
    dir: str = "outputs/feature_cache"
    # Read/write the on-disk cache. Off = always recompute in-process (the pre-existing behaviour).
    reuse_disk: bool = True


@dataclass
class WandBConfig:
    # Opt-in Weights & Biases logging of per-epoch train/val curves. Mirrors the System-1 trainer.
    enable: bool = False
    mode: str = "online"            # "online" | "offline" | "disabled"
    project: str = "RoboMLT_system2"
    entity: str = ""                # "" → default entity
    log_interval: int = 50          # upload train losses every N optimizer steps


@dataclass
class CompletionGateTrainConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    batch_size: int = 128
    num_workers: int = 4
    # Optimizer steps (Table II: 60k). Training runs whole epochs until this many steps, evaluating
    # and checkpointing at the end of each epoch and once more at the final step.
    max_steps: int = 60000
    output_dir: str = "./outputs/completion_gate"
    run_name: Optional[str] = None
    gpu: int = 0
    seed: int = 42
    # Also snapshot the gate every N epochs (0 disables). A rolling ``completion_gate_last.pth`` is
    # always written each epoch regardless — the only selectable checkpoint in train-only mode.
    save_interval: int = 0


# ────────────────────────────── Config loading ───────────────────────────────

def _set_nested(d: dict, key_path: str, value):
    """Set a dot-notation key inside a nested dict (mutates in place)."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _apply_legacy_keys(raw: dict) -> None:
    """Rename pre-release keys in place (with a warning) so old YAMLs keep loading."""
    for section, mapping in _LEGACY_KEYS.items():
        sec = raw.get(section) or {}
        for old, new in mapping.items():
            if old in sec:
                logger.warning("Config key '%s.%s' is deprecated; use '%s.%s'.", section, old, section, new)
                sec.setdefault(new, sec.pop(old))
    model = raw.get("model") or {}
    if "use_cross_attention" in model:
        legacy = model.pop("use_cross_attention")
        model.setdefault("fusion", "legacy_cross_attention" if legacy else "concat")
    if "num_epochs" in raw:
        raise ValueError("'num_epochs' was replaced by 'max_steps' (Table II: 60000 steps).")
    if "max_episode_index" in (raw.get("dataset") or {}):
        raise ValueError("'dataset.max_episode_index' was removed: release datasets no longer "
                         "contain duplicated episodes.")


def _to_config(raw: dict) -> CompletionGateTrainConfig:
    """Convert a raw dict (from YAML) into a CompletionGateTrainConfig."""
    _apply_legacy_keys(raw)
    return CompletionGateTrainConfig(
        dataset=DatasetConfig(**{k: v for k, v in (raw.pop("dataset", {}) or {}).items()}),
        sampler=SamplerConfig(**{k: v for k, v in (raw.pop("sampler", {}) or {}).items()}),
        model=ModelConfig(**{k: v for k, v in (raw.pop("model", {}) or {}).items()}),
        optimizer=OptimizerConfig(**{k: v for k, v in (raw.pop("optimizer", {}) or {}).items()}),
        loss=LossConfig(**{k: v for k, v in (raw.pop("loss", {}) or {}).items()}),
        cache=CacheConfig(**{k: v for k, v in (raw.pop("cache", {}) or {}).items()}),
        wandb=WandBConfig(**{k: v for k, v in (raw.pop("wandb", {}) or {}).items()}),
        **{k: v for k, v in raw.items() if k in CompletionGateTrainConfig.__dataclass_fields__},
    )


def load_config(yaml_path: str, overrides: List[str]) -> CompletionGateTrainConfig:
    """Load YAML config and apply ``key=value`` CLI overrides."""
    with open(yaml_path) as f:
        raw: dict = yaml.safe_load(f) or {}

    for override in overrides:
        key, _, val_str = override.partition("=")
        try:
            val = yaml.safe_load(val_str)
        except Exception:  # noqa: BLE001
            val = val_str
        _set_nested(raw, key, val)

    return _to_config(raw)


# ───────────────────────────── Training helpers ──────────────────────────────

def build_run_name(cfg: CompletionGateTrainConfig) -> str:
    """Build a descriptive, self-documenting run name so checkpoints from different
    configs/datasets don't overwrite each other. Trailing timestamp keeps re-runs unique.

    e.g. blood_gas_analysis_cross_attention_seg16x8_lr1_lc1_h6o0s19_1022_153012
    """
    dataset = os.path.basename(os.path.normpath(cfg.dataset.repo_id))
    parts = [dataset, cfg.model.fusion]
    if cfg.dataset.random_skip_range > 0:
        parts.append(f"rskip{cfg.dataset.random_skip_range}")
    if cfg.sampler.use_segment_sampler:
        parts.append(f"seg{cfg.sampler.segs_per_batch}x{cfg.sampler.frames_per_seg}")
    if cfg.dataset.max_episodes is not None:
        parts.append(f"ep{cfg.dataset.max_episodes}")
    parts += [
        f"lr{cfg.loss.lambda_rec:g}",
        f"lc{cfg.loss.lambda_completion:g}",
        f"h{cfg.dataset.history_len}o{cfg.dataset.prediction_offset}s{cfg.dataset.history_skip_frame}",
        datetime.now().strftime("%m%d_%H%M%S"),
    ]
    return "_".join(parts)


# Filename of the sidecar written next to the checkpoint(s); read back at inference to rebuild the
# backbone + gate without hand-copying these fields into the inference YAML.
GATE_SIDECAR_NAME = "completion_gate_config.yaml"


def build_gate_sidecar(cfg: CompletionGateTrainConfig, skill_texts: Optional[List[str]] = None,
                       trainable_params: Optional[int] = None,
                       dataset_fps: Optional[float] = None) -> dict:
    """The minimal, human-readable set of fields needed to *reconstruct* the gate at inference:
    the backbone-build params plus the gate-architecture flags. Written as YAML next to the .pth.

    ``skill_texts`` records the subtask strings the heads were actually trained on, so deployment can
    flag a skill library that has since been reworded.
    """
    return {
        "repo_id": cfg.dataset.repo_id,
        "skill_texts": list(skill_texts or []),
        "camera_names": list(cfg.dataset.camera_names),
        "history_len": cfg.dataset.history_len,
        "prediction_offset": cfg.dataset.prediction_offset,
        "history_skip_frame": cfg.dataset.history_skip_frame,
        "random_skip_range": cfg.dataset.random_skip_range,
        # Lets deployment compare the trained frame spacing with K / f_ctrl.
        "dataset_fps": dataset_fps,
        # Label-construction params: the eval video rebuilds the GT completion curve from these, so
        # they must round-trip or the overlay won't match what the head was trained on.
        "completion_window": cfg.dataset.completion_window,
        "completion_at_episode_end": cfg.dataset.completion_at_episode_end,
        "model": {
            "fusion": cfg.model.fusion,
            "temporal_layers": cfg.model.temporal_layers,
            "temporal_heads": cfg.model.temporal_heads,
            "cross_attn_layers": cfg.model.cross_attn_layers,
            "cross_attn_heads": cfg.model.cross_attn_heads,
        },
        "trainable_params": trainable_params,
    }


def build_loaders(cfg: CompletionGateTrainConfig, full_dataset, val_dataset=None,
                  recovery_dataset=None):
    """Build train/val datasets + loaders.

    Two modes. Normally the episodes of ``full_dataset`` are split by ``train_ratio``. If
    ``dataset.val_repo_id`` is set, ``val_dataset`` (a second LeRobotDataset) supplies the val split
    instead and *every* episode of ``full_dataset`` trains — a real cross-capture generalization
    test rather than held-out episodes of the same session.

    ``recovery_dataset`` (the intervention / failure corpus) is split by ``train_ratio`` with the
    same seed and its parts are concatenated onto the train / val splits.
    """
    cross_val = val_dataset is not None
    min_len = int(cfg.dataset.min_episode_len or 0)
    keep_back = bool(cfg.dataset.include_back_episodes)

    def _back_episodes(dset) -> set:
        """Episode indices with ANY back_event positive, read decode-free from the inline column.
        Empty when the column is absent or ``include_back_episodes`` is off."""
        if not keep_back:
            return set()
        hf = getattr(dset, "hf_dataset", None)
        if hf is None or "back_event" not in getattr(hf, "column_names", []):
            logger.warning("include_back_episodes=true but no 'back_event' column found — ignoring.")
            return set()
        vals = np.asarray(hf["back_event"]).reshape(len(hf), -1)[:, 0]
        out = set()
        for ep in range(dset.meta.total_episodes):
            s, e = _get_ep_bounds(dset.meta, ep)
            if bool(vals[s:e].any()):
                out.add(int(ep))
        return out

    def _keep_episodes(dset) -> List[int]:
        """Episodes kept by the UNION filter: length strictly > ``min_len`` OR (when enabled) the
        episode contains a back_event frame. All episodes when every filter is off."""
        back_set = _back_episodes(dset)
        keep, n_long, n_back_only = [], 0, 0
        for ep in range(dset.meta.total_episodes):
            s, e = _get_ep_bounds(dset.meta, ep)
            is_long = min_len <= 0 or (e - s) > min_len
            if is_long:
                n_long += 1
                keep.append(ep)
            elif ep in back_set:
                n_back_only += 1
                keep.append(ep)
        if min_len > 0 or keep_back:
            logger.info("Episode filter: kept %d / %d = %d long (len>%d) ∪ %d back-only clips.",
                        len(keep), dset.meta.total_episodes, n_long, min_len, n_back_only)
        return keep

    all_indices = np.array(_keep_episodes(full_dataset), dtype=int)
    np.random.seed(cfg.dataset.episode_subset_seed)
    np.random.shuffle(all_indices)
    # Optionally keep only a random subset of episodes (the dataset can be huge). Since the indices
    # are already shuffled, truncating gives a uniform random sample; the train/val split below is
    # then taken over that subset, so both splits shrink proportionally.
    max_eps = cfg.dataset.max_episodes
    pool = len(all_indices)
    if max_eps is not None and 0 < max_eps < pool:
        all_indices = all_indices[:max_eps]
        logger.info("Episode sub-sampling ON: using %d / %d episodes (seed %d).",
                    max_eps, pool, cfg.dataset.episode_subset_seed)
    else:
        logger.info("Using all %d episodes in pool.", pool)
    n_eps = len(all_indices)
    if cross_val:
        # The val split comes from another dataset, so nothing is held out here.
        train_eps, val_eps = all_indices.tolist(), []
    else:
        # train_ratio >= 1.0 (or a split that leaves no held-out episodes) → train-only mode: every
        # episode goes to the train split and validation is skipped entirely downstream.
        split = min(n_eps, int(round(n_eps * cfg.dataset.train_ratio)))
        train_eps, val_eps = all_indices[:split].tolist(), all_indices[split:].tolist()
    if cross_val:
        logger.info("Cross-dataset validation: %d train episodes from %s; val from %s episodes %s.",
                    len(train_eps), cfg.dataset.repo_id, cfg.dataset.val_repo_id,
                    cfg.dataset.val_episodes if cfg.dataset.val_episodes is not None else "(all)")
    elif not val_eps:
        logger.info("Train-only mode: %d train episodes, no validation split.", len(train_eps))
    else:
        logger.info("Episode split: %d train / %d val.", len(train_eps), len(val_eps))

    common = dict(
        camera_names=cfg.dataset.camera_names,
        history_len=cfg.dataset.history_len,
        prediction_offset=cfg.dataset.prediction_offset,
        history_skip_frame=cfg.dataset.history_skip_frame,
        use_command_in_meta=cfg.dataset.use_command_in_meta,
        back_annotations_path=cfg.dataset.back_annotations,
        back_from_column=cfg.dataset.back_from_column,
        min_segment_len=cfg.dataset.min_segment_len,
        completion_window=cfg.dataset.completion_window,
        completion_at_episode_end=cfg.dataset.completion_at_episode_end,
        random_skip_range=cfg.dataset.random_skip_range,
    )
    # randomize the stride only on the train split; val stays deterministic for comparable metrics.
    train_ds = CompletionGateDataset(full_dataset, subset_episodes=train_eps,
                                     randomize_skip=cfg.dataset.random_skip_range > 0, **common)
    if cross_val:
        val_subset = cfg.dataset.val_episodes
        if val_subset is None and (min_len > 0 or keep_back):
            val_subset = _keep_episodes(val_dataset)
        val_ds = CompletionGateDataset(val_dataset, subset_episodes=val_subset,
                                       randomize_skip=False, **common)
    else:
        val_ds = (CompletionGateDataset(full_dataset, subset_episodes=val_eps, randomize_skip=False,
                                        **common) if val_eps else None)

    if recovery_dataset is not None:
        rec_eps = np.arange(recovery_dataset.meta.total_episodes)
        np.random.RandomState(cfg.dataset.episode_subset_seed).shuffle(rec_eps)
        split = min(len(rec_eps), int(round(len(rec_eps) * cfg.dataset.train_ratio)))
        rec_train, rec_val = rec_eps[:split].tolist(), rec_eps[split:].tolist()
        logger.info("Recovery corpus %s: %d train / %d val episodes.",
                    cfg.dataset.recovery_repo_id, len(rec_train), len(rec_val))
        # A recovery episode ends in the disturbance, not in a finished skill: its final run must
        # not become completion positives on the frames the recovery head is trained to flag.
        rec_common = {**common, "completion_at_episode_end": False}
        rec_train_ds = CompletionGateDataset(recovery_dataset, subset_episodes=rec_train,
                                             randomize_skip=cfg.dataset.random_skip_range > 0,
                                             **rec_common)
        train_ds = ConcatGateDataset([train_ds, rec_train_ds])
        if rec_val:
            rec_val_ds = CompletionGateDataset(recovery_dataset, subset_episodes=rec_val,
                                               randomize_skip=False, **rec_common)
            val_ds = ConcatGateDataset([p for p in (val_ds, rec_val_ds) if p is not None])

    if cfg.sampler.use_segment_sampler:
        # Group each batch into a few segments × several frames so the rare positives are present
        # in every batch. batch_size = segs_per_batch * frames_per_seg.
        sampler = SegmentBatchSampler(
            train_ds, segs_per_batch=cfg.sampler.segs_per_batch,
            frames_per_seg=cfg.sampler.frames_per_seg, shuffle=True,
        )
        train_loader = make_completion_gate_loader(
            train_ds, cfg.batch_size, True, cfg.num_workers, batch_sampler=sampler)
        logger.info("Segment sampler ON: %d segs/batch × %d frames = batch %d, %d batches/epoch.",
                    cfg.sampler.segs_per_batch, cfg.sampler.frames_per_seg,
                    cfg.sampler.segs_per_batch * cfg.sampler.frames_per_seg, len(sampler))
    else:
        train_loader = make_completion_gate_loader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_loader = (make_completion_gate_loader(val_ds, cfg.batch_size, False, cfg.num_workers)
                  if val_ds is not None else None)
    return train_ds, val_ds, train_loader, val_loader

# ───────────────────── Frame-level frozen-SigLIP cache (trunk-free) ───────────

def frame_cache_collate_fn(batch):
    n_comp = len(batch[0][0])
    feats = tuple(torch.stack([b[0][c] for b in batch], dim=0) for c in range(n_comp))
    skills = [b[1] for b in batch]
    back = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    complete = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    seg_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return feats, skills, back, complete, seg_ids

class _FrameImageDataset(Dataset):
    """Serves one multi-camera frame ``[Cams, C, H, W]`` per global frame index (for the encode sweep)."""

    def __init__(self, lerobot_ds, frame_indices: List[int], camera_names: List[str]) -> None:
        self.dset = lerobot_ds
        self.frame_indices = frame_indices
        self.camera_names = camera_names

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, i):
        data = self.dset[self.frame_indices[i]]
        return torch.stack([data[cam] for cam in self.camera_names], dim=0)


class FrameGatherDataset(Dataset):
    """History windows gathered on demand from a per-global-frame SigLIP feature cache.

    Keying the cache by *frame* (rather than by sample) lets ``__getitem__`` redraw the stride
    every epoch and simply re-index — so ``random_skip_range`` augmentation is real — while encoding
    each frame exactly once instead of once per history window it appears in (~4x fewer SigLIP
    forwards on a typical config).

    Labels come from :meth:`CompletionGateDataset.labels_for`, so they are constructed by the same
    code as the uncached path. ``samples`` / ``frame_to_seg`` are re-exposed for
    :class:`SegmentBatchSampler`.
    """

    def __init__(self, base_ds: CompletionGateDataset, frame_feats, index_map: dict) -> None:
        self.base = base_ds
        self.samples = base_ds.samples
        self.frame_to_seg = base_ds.frame_to_seg
        self.frame_feats = frame_feats      # [F, Cams, D]; may be a read-only numpy memmap
        self.index_map = index_map          # global frame index -> row in frame_feats
        self.num_pad_frames = 0             # windows are gathered whole; nothing is padded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        _, curr = self.samples[i]
        rows = [self.index_map[f] for f in self.base.history_indices(curr, self.base.draw_stride())]
        feats = self.frame_feats[rows]      # [T, Cams, D] — fancy indexing already copies
        if not torch.is_tensor(feats):
            feats = torch.from_numpy(np.ascontiguousarray(feats))
        skill, back, complete, seg_id = self.base.labels_for(i)
        return (feats,), skill, back, complete, seg_id


# ── on-disk cache: keyed so parallel sweep processes share one copy ──────────────

def _cache_key(kind: str, repo_id: str, camera_names: List[str], history_len: int,
               frame_indices: List[int], model_name: str) -> str:
    """Short digest of everything that changes the cached tensor.

    ``camera_names`` is hashed **in order** (not sorted): it is the Cams axis of the stored tensor,
    so a reordering is a genuinely different cache, not the same one under another name.
    """
    h = hashlib.sha1()
    for part in (kind, str(repo_id), "\x1f".join(camera_names), str(history_len), str(model_name)):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    h.update(np.asarray(frame_indices, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _cache_load(cache_dir: str, key: str):
    """Return the memmapped array if a *complete* cache exists, else None.

    The manifest is written last (see :func:`_cache_store`), so its presence certifies that
    ``feats.npy`` is fully flushed — a reader can never observe a half-written array.
    """
    d = os.path.join(cache_dir, key)
    feats_p, man_p = os.path.join(d, "feats.npy"), os.path.join(d, "manifest.yaml")
    if not (os.path.isfile(feats_p) and os.path.isfile(man_p)):
        return None
    try:
        with open(man_p) as f:
            man = yaml.safe_load(f) or {}
        # mmap so N concurrent processes share one set of pages instead of N private copies.
        arr = np.load(feats_p, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 — a corrupt cache must never be fatal
        logger.warning("Ignoring unreadable cache %s: %s", d, exc)
        return None
    if int(man.get("num_frames", -1)) != arr.shape[0]:
        logger.warning("Cache %s is inconsistent (manifest %s rows vs array %d) — recomputing.",
                       d, man.get("num_frames"), arr.shape[0])
        return None
    return arr, man


def _cache_store(cache_dir: str, key: str, arr: np.ndarray, manifest: dict) -> str:
    """Write the array then the manifest, each atomically (tmp file + ``os.replace``)."""
    d = os.path.join(cache_dir, key)
    os.makedirs(d, exist_ok=True)
    feats_p, man_p = os.path.join(d, "feats.npy"), os.path.join(d, "manifest.yaml")
    tmp = f"{feats_p}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)          # file object → numpy won't append its own .npy suffix
    os.replace(tmp, feats_p)
    tmp = f"{man_p}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, man_p)       # written last: its presence means feats.npy is complete
    return d


def _episode_frame_indices(base_ds) -> List[int]:
    """Every global frame index of every episode ``base_ds`` draws samples from."""
    frame_indices: List[int] = []
    for ep in sorted({ep for ep, _ in base_ds.samples}):
        ep_start, ep_end = _get_ep_bounds(base_ds.dset.meta, ep)
        frame_indices.extend(range(ep_start, ep_end))
    return frame_indices


def precompute_frame_features(gate, base_ds, device, cfg):
    """Per-global-frame frozen SigLIP features, from disk when available.

    ``encode_siglip_frames`` is reused as-is by feeding a *chunk of frames* along its time axis: it
    only zero-pads when the axis is shorter than ``history_len + 1``, so a chunk at least that long
    comes back unpadded and its per-frame features are exactly what we want.

    Returns ``(frame_feats [F, Cams, D] numpy, index_map)``. The array may be a read-only memmap;
    :class:`FrameGatherDataset` copies each gathered window, so that is transparent downstream.
    """
    frame_indices = _episode_frame_indices(base_ds)
    index_map = {f: r for r, f in enumerate(frame_indices)}
    model_name = type(getattr(gate.backbone, "siglip_model", None)).__name__
    key = _cache_key("siglip_frames", base_ds.dset.repo_id, list(base_ds.camera_names),
                     int(base_ds.history_len), frame_indices, model_name)

    if cfg.cache.reuse_disk:
        hit = _cache_load(cfg.cache.dir, key)
        if hit is not None:
            arr, _ = hit
            logger.info("Frame cache HIT [%s]: %s loaded from disk (mmap, %.1f MB shared).",
                        key, tuple(arr.shape), arr.nbytes / 1e6)
            return arr, index_map
        logger.info("Frame cache MISS [%s under %s] — computing.", key, cfg.cache.dir)

    # Chunk must clear history_len + 1 or encode_siglip_frames would pad and shift the rows.
    min_chunk = int(getattr(gate.backbone, "history_len", base_ds.history_len)) + 1
    chunk = max(int(cfg.cache.precompute_batch_size), min_chunk)
    loader = DataLoader(
        _FrameImageDataset(base_ds.dset, frame_indices, base_ds.camera_names),
        batch_size=chunk, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    gate.eval()
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for frames in tqdm(loader, desc="precompute-frame-cache"):
            # [N, Cams, C, H, W] -> [1, N, Cams, C, H, W]: N rides the time axis.
            imgs = frames.to(device, non_blocking=True).unsqueeze(0)
            feats, num_pad = gate.backbone.encode_siglip_frames(imgs)
            if num_pad:
                feats = feats[:, num_pad:]      # only possible on a short final chunk
            out.append(feats[0].detach().float().cpu())
    frame_feats = torch.cat(out, dim=0).numpy()         # [F, Cams, D]
    assert frame_feats.shape[0] == len(frame_indices), (
        f"frame cache built {frame_feats.shape[0]} rows for {len(frame_indices)} frames")
    logger.info("Frame cache: %d frames x %d cams -> %s | %.1f MB (%d samples reference it).",
                len(frame_indices), frame_feats.shape[1], tuple(frame_feats.shape),
                frame_feats.nbytes / 1e6, len(base_ds.samples))

    if cfg.cache.reuse_disk:
        d = _cache_store(cfg.cache.dir, key, frame_feats, {
            "kind": "siglip_frames", "repo_id": base_ds.dset.repo_id,
            "camera_names": list(base_ds.camera_names), "history_len": int(base_ds.history_len),
            "siglip_model": model_name, "num_frames": int(frame_feats.shape[0]),
            "shape": list(frame_feats.shape), "dtype": str(frame_feats.dtype),
            "episodes": sorted({int(ep) for ep, _ in base_ds.samples}),
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        logger.info("Frame cache stored -> %s", d)
    return frame_feats, index_map


def _cached(gate, ds, device, cfg):
    """Frame-cached view of ``ds``; a :class:`ConcatGateDataset` is cached part by part."""
    if isinstance(ds, ConcatGateDataset):
        return ConcatGateDataset([_cached(gate, part, device, cfg) for part in ds.parts])
    feats, index_map = precompute_frame_features(gate, ds, device, cfg)
    return FrameGatherDataset(ds, feats, index_map)


def build_frame_cached_loaders(gate, train_ds, val_ds, device, cfg):
    """Frame-cached loaders. num_workers=0: indexing in-RAM tensors is trivial, so workers would
    only add IPC/copy overhead."""
    common = dict(num_workers=0, pin_memory=True, collate_fn=frame_cache_collate_fn)
    train_cached = _cached(gate, train_ds, device, cfg)
    if cfg.sampler.use_segment_sampler:
        sampler = SegmentBatchSampler(
            train_cached, segs_per_batch=cfg.sampler.segs_per_batch,
            frames_per_seg=cfg.sampler.frames_per_seg, shuffle=True)
        train_loader = DataLoader(train_cached, batch_sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_cached, batch_size=cfg.batch_size, shuffle=True, **common)
    if val_ds is None:
        return train_loader, None
    val_cached = _cached(gate, val_ds, device, cfg)
    val_loader = DataLoader(val_cached, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader


def dataset_skill_texts(dataset) -> List[str]:
    """Distinct subtask strings the gate will actually be trained on (segment order-independent)."""
    return sorted({seg.skill_text for seg in dataset.segments if seg.skill_text})


def check_skill_library_alignment(skill_texts: List[str], library_path: Optional[str]) -> None:
    """Warn when the deployed skill library's ``canonical_instruction`` strings don't match the
    dataset's ``meta/subtasks`` text the heads were trained on.

    The gate conditions on a frozen SigLIP embedding of the skill string, so a reworded instruction
    lands somewhere else in text space and silently degrades the heads. Nothing in the repo keeps the
    two files in sync, so this is a pure drift check. It is a *warning*, not an error: feeding an
    unseen skill is a legitimate open-set generalization test — you just want to know you're doing it.
    """
    if not library_path:
        return
    try:
        from high_level_model.planning.skill_library import load_skill_library
        library = load_skill_library(library_path)
    except Exception as exc:  # noqa: BLE001 — never block training on a bookkeeping check
        logger.warning("Could not load skill library %s for the alignment check: %s", library_path, exc)
        return
    lib_texts = set(library.instructions())
    ds_texts = set(skill_texts)
    unseen = sorted(lib_texts - ds_texts)
    untested = sorted(ds_texts - lib_texts)
    if unseen:
        logger.warning("Skill-library instructions NOT present in the training data (open-set at "
                       "deploy time — verify this is intentional): %s", unseen)
    if untested:
        logger.warning("Trained subtask strings absent from the skill library (never reachable at "
                       "deploy time): %s", untested)
    if not unseen and not untested:
        logger.info("Skill-library / dataset subtask texts match exactly (%d skills).", len(lib_texts))


def compute_pos_weight(dataset) -> float:
    pos = sum(1 for _, f in dataset.samples if f in dataset.back_frames)
    neg = len(dataset.samples) - pos
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


def compute_completion_pos_weight(dataset) -> float:
    """pos_weight for the rare completion positives (last frames of completed segments)."""
    pos = sum(1 for _, f in dataset.samples if f in dataset.completion_frames)
    neg = len(dataset.samples) - pos
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


def run_epoch(gate, loader, optimizer, device, cfg, bce, bce_comp, train: bool,
              cached: bool = False, wandb_run=None, log_interval: int = 50, step_counter=None,
              scheduler=None):
    gate.train() if train else gate.eval()
    tot, tot_back, tot_comp, n = 0.0, 0.0, 0.0, 0
    tp = fp = fn = 0
    ctp = cfp = cfn = 0
    num_pad = getattr(loader.dataset, "num_pad_frames", 0)
    torch.set_grad_enabled(train)
    for batch in tqdm(loader, desc="train" if train else "val"):
        if cached:
            feats, skills, back, complete, seg_ids = batch
            feats = tuple(f.to(device, non_blocking=True) for f in feats)
        else:
            images, skills, back, complete, seg_ids = batch
            images = images.to(device, non_blocking=True)
        back = back.to(device)
        complete = complete.to(device)
        seg_ids = seg_ids.to(device)
        if cached:
            comp_logit, back_logit = gate.forward_from_cached(feats, skills, num_pad_frames=num_pad)
        else:
            comp_logit, back_logit = gate(images, skills)
        loss_back = bce(back_logit, back)
        loss_comp = bce_comp(comp_logit, complete)
        loss = cfg.loss.lambda_rec * loss_back + cfg.loss.lambda_completion * loss_comp
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # Per-step W&B logging on a fixed interval (global optimizer step is the x-axis, so the
            # later per-epoch logs stay monotonically increasing on the same axis).
            if step_counter is not None:
                step_counter[0] += 1
                if wandb_run is not None and step_counter[0] % max(1, log_interval) == 0:
                    wandb_run.log({
                        "train_step/loss": loss.item(),
                        "train_step/back": loss_back.item(),
                        "train_step/comp": loss_comp.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }, step=step_counter[0])
        tot += loss.item(); tot_back += loss_back.item(); tot_comp += loss_comp.item(); n += 1
        pred_back = (torch.sigmoid(back_logit) >= 0.5).float()
        tp += int(((pred_back == 1) & (back == 1)).sum())
        fp += int(((pred_back == 1) & (back == 0)).sum())
        fn += int(((pred_back == 0) & (back == 1)).sum())
        pred_comp = (torch.sigmoid(comp_logit) >= 0.5).float()
        ctp += int(((pred_comp == 1) & (complete == 1)).sum())
        cfp += int(((pred_comp == 1) & (complete == 0)).sum())
        cfn += int(((pred_comp == 0) & (complete == 1)).sum())
        if train and step_counter is not None and step_counter[0] >= cfg.max_steps:
            break
    torch.set_grad_enabled(True)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    cprec = ctp / (ctp + cfp) if (ctp + cfp) else 0.0
    crec = ctp / (ctp + cfn) if (ctp + cfn) else 0.0
    return (tot / max(1, n), tot_back / max(1, n), tot_comp / max(1, n), prec, rec, cprec, crec)


def build_scheduler(optimizer, cfg: CompletionGateTrainConfig):
    """Linear warm-up then cosine decay from ``lr`` to ``min_lr`` over ``max_steps`` (Table II)."""
    sched = cfg.optimizer.schedule
    if sched == "constant":
        return None
    if sched != "cosine":
        raise ValueError(f"optimizer.schedule must be 'cosine' or 'constant', got '{sched}'")
    warmup = max(0, int(cfg.optimizer.warmup_steps))
    total = max(1, int(cfg.max_steps))
    floor = cfg.optimizer.min_lr / cfg.optimizer.lr if cfg.optimizer.lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────── Main train ──────────────────────────────────

@torch.no_grad()
def collect_val_scores(gate, loader, device, cfg, cached: bool):
    """``(completion_probs, completion_labels)`` over a whole loader — the inputs to a
    threshold-free score (see ``eval.gate_metrics.completion_ap``). Batched and cheap enough to run
    every epoch, unlike the sequential closed-loop metric."""
    gate.eval()
    num_pad = getattr(loader.dataset, "num_pad_frames", 0)
    probs: List[float] = []
    labels: List[float] = []
    for batch in loader:
        if cached:
            feats, skills, _, complete, _ = batch
            feats = tuple(f.to(device, non_blocking=True) for f in feats)
        else:
            images, skills, _, complete, _ = batch
            images = images.to(device, non_blocking=True)
        if cached:
            comp_logit, _ = gate.forward_from_cached(feats, skills, num_pad_frames=num_pad)
        else:
            comp_logit, _ = gate(images, skills)
        probs.extend(torch.sigmoid(comp_logit).float().cpu().tolist())
        labels.extend(complete.float().cpu().tolist())
    return probs, labels


def train(cfg: CompletionGateTrainConfig, on_epoch_end=None) -> str:
    """Train the gate; returns the run directory.

    ``on_epoch_end(ctx)`` is invoked after each epoch with a dict carrying the metrics plus live
    handles (``gate``, ``val_loader``, ``device``, ``cached``). The sweep uses it to log per-epoch
    curves without duplicating the training loop.
    """
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_name = cfg.run_name or build_run_name(cfg)
    # Group every run started on the same calendar day under a YYYYMMDD parent folder, so
    # checkpoints live at {output_dir}/{YYYYMMDD}/{run_name}/ and the eval video lands in
    # .../{run_name}/videos/ (derived from the checkpoint dir).
    date_dir = datetime.now().strftime("%Y%m%d")
    run_dir = os.path.join(cfg.output_dir, date_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run: %s/%s", date_dir, run_name)
    logger.info("Checkpoints -> %s", run_dir)

    wandb_run = None
    if cfg.wandb.enable:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity or None,
            name=run_name, mode=cfg.wandb.mode, dir=run_dir,
            config=dataclasses.asdict(cfg),
        )
        logger.info("W&B logging ON -> project=%s run=%s", cfg.wandb.project, run_name)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    image_transforms = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _tol = {} if cfg.dataset.tolerance_s is None else {"tolerance_s": cfg.dataset.tolerance_s}
    full_dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, image_transforms=image_transforms, **_tol)
    val_dataset = (LeRobotDataset(repo_id=cfg.dataset.val_repo_id, image_transforms=image_transforms, **_tol)
                   if cfg.dataset.val_repo_id else None)
    recovery_dataset = (LeRobotDataset(repo_id=cfg.dataset.recovery_repo_id,
                                       image_transforms=image_transforms, **_tol)
                        if cfg.dataset.recovery_repo_id else None)

    train_ds, val_ds, train_loader, val_loader = build_loaders(
        cfg, full_dataset, val_dataset, recovery_dataset)

    skill_texts = dataset_skill_texts(train_ds)
    check_skill_library_alignment(skill_texts, cfg.dataset.skill_library)

    backbone = build_siglip_encoder(
        device, cfg.dataset.history_len, freeze_siglip=cfg.model.freeze_siglip,
    ).to(device)

    gate = CompletionGate(
        backbone, freeze_siglip=cfg.model.freeze_siglip, fusion=cfg.model.fusion,
        temporal_layers=cfg.model.temporal_layers, temporal_heads=cfg.model.temporal_heads,
        cross_attn_layers=cfg.model.cross_attn_layers, cross_attn_heads=cfg.model.cross_attn_heads,
        cache_text_embeddings=cfg.cache.text_embeddings,
    ).to(device)
    trainable = [p for p in gate.parameters() if p.requires_grad]
    n_trainable = gate.num_trainable_parameters()
    logger.info("Trainable params (trunk + heads): %d (%.3fM)", n_trainable, n_trainable / 1e6)
    optimizer = optim.AdamW(trainable, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)

    # The sidecar records the exact skill strings the heads were trained on (deployment compares
    # its skill library against them) and the trainable parameter count.
    sidecar_path = os.path.join(run_dir, GATE_SIDECAR_NAME)
    with open(sidecar_path, "w") as f:
        yaml.safe_dump(build_gate_sidecar(cfg, skill_texts, n_trainable,
                                          getattr(full_dataset.meta, "fps", None)), f,
                       sort_keys=False, allow_unicode=True)
    logger.info("Wrote gate sidecar -> %s", sidecar_path)

    cache_stage = gate.cacheable_stage  # "frames" (SigLIP frozen) or None (SigLIP trains)
    use_feature_cache = cfg.cache.features and cache_stage is not None
    if cfg.cache.features and cache_stage is None:
        logger.warning("cache.features requested but the SigLIP tower is trainable "
                       "(freeze_siglip=False) — its features change every epoch; "
                       "running the normal (uncached) path instead.")
    if use_feature_cache:
        logger.info("Feature cache ON: frozen SigLIP → caching one feature per GLOBAL FRAME; "
                    "history windows (and their random stride) are re-gathered every epoch, so "
                    "stride augmentation actually varies.")
        train_loader, val_loader = build_frame_cached_loaders(gate, train_ds, val_ds, device, cfg)

    pos_weight = torch.tensor(
        [cfg.loss.rec_pos_weight or compute_pos_weight(train_ds)], device=device)
    comp_pos_weight = torch.tensor(
        [cfg.loss.completion_pos_weight or compute_completion_pos_weight(train_ds)], device=device)
    logger.info("recovery pos_weight (w_r) = %.2f | completion pos_weight (w_c) = %.2f",
                pos_weight.item(), comp_pos_weight.item())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_comp = nn.BCEWithLogitsLoss(pos_weight=comp_pos_weight)

    # Train-only mode (no val split): selection + collapse monitoring fall back to the train metrics,
    # which are the only ones available. "val" below is an alias for whichever split we score on.
    has_val = val_loader is not None
    metric_split = "val" if has_val else "train"

    def save_gate(path: str, epoch: int, metrics) -> None:
        torch.save({"epoch": epoch, "state_dict": gate.state_dict(),
                    f"{metric_split}_completion_bce": metrics[2], "run_name": run_name,
                    "config": dataclasses.asdict(cfg)}, path)

    keys = ["loss", "back", "comp", "back_prec", "back_rec", "comp_prec", "comp_rec"]
    best_val = float("inf")
    step_counter = [0]   # global optimizer-step count, shared across epochs (W&B x-axis)
    num_epochs = math.ceil(cfg.max_steps / max(1, len(train_loader)))
    logger.info("Training for %d steps (%d batches/epoch -> %d epochs).",
                cfg.max_steps, len(train_loader), num_epochs)
    for epoch in range(num_epochs):
        tr = run_epoch(gate, train_loader, optimizer, device, cfg, bce, bce_comp, True,
                       cached=use_feature_cache, wandb_run=wandb_run,
                       log_interval=cfg.wandb.log_interval, step_counter=step_counter,
                       scheduler=scheduler)
        # tuple: (loss, back, comp, back_prec, back_rec, comp_prec, comp_rec)
        va = (run_epoch(gate, val_loader, optimizer, device, cfg, bce, bce_comp, False,
                        cached=use_feature_cache) if has_val else None)
        sel = va if has_val else tr   # metrics used for selection
        if has_val:
            logger.info(
                "Epoch %d | train loss %.4f (back %.4f comp %.4f) | val (back %.4f comp %.4f) | "
                "back P/R %.2f/%.2f | comp P/R %.2f/%.2f",
                epoch, tr[0], tr[1], tr[2], va[1], va[2], va[3], va[4], va[5], va[6],
            )
        else:
            logger.info(
                "Epoch %d | train loss %.4f (back %.4f comp %.4f) | "
                "back P/R %.2f/%.2f | comp P/R %.2f/%.2f",
                epoch, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], tr[6],
            )
        if wandb_run is not None:
            # End-of-epoch aggregates logged on the same global-step axis as the per-step train logs,
            # so the step stays monotonically increasing (step_counter[0] only grows).
            log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            log.update({f"train/{k}": v for k, v in zip(keys, tr)})
            if has_val:
                log.update({f"val/{k}": v for k, v in zip(keys, va)})
            wandb_run.log(log, step=step_counter[0])

        # Always keep a rolling "last" snapshot (the only checkpoint in train-only mode if you
        # don't trust an intermediate metric), plus optional every-N-epoch snapshots.
        save_gate(os.path.join(run_dir, "completion_gate_last.pth"), epoch, sel)
        if cfg.save_interval and (epoch + 1) % cfg.save_interval == 0:
            save_gate(os.path.join(run_dir, f"completion_gate_ep{epoch:04d}.pth"), epoch, sel)

        if sel[2] < best_val:  # select by completion-head BCE (the deployed advance signal)
            best_val = sel[2]
            path = os.path.join(run_dir, "completion_gate_best.pth")
            save_gate(path, epoch, sel)
            logger.info("Saved %s (%s completion BCE %.4f)", path, metric_split, best_val)
            if wandb_run is not None:
                wandb_run.summary[f"best_{metric_split}_completion_bce"] = best_val
                wandb_run.summary["best_epoch"] = epoch

        if on_epoch_end is not None:
            on_epoch_end({
                "epoch": epoch, "run_dir": run_dir, "keys": keys,
                "train": dict(zip(keys, tr)), "val": dict(zip(keys, va)) if has_val else None,
                "gate": gate, "val_loader": val_loader, "val_ds": val_ds,
                "device": device, "cached": use_feature_cache, "cfg": cfg,
            })
        if step_counter[0] >= cfg.max_steps:
            break

    if wandb_run is not None:
        wandb_run.finish()
    return run_dir

# ─────────────────────────────────── CLI ─────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print(
            "Usage: python -m high_level_model.training.train_completion_gate "
            "<config.yaml> [key=value ...]\n"
            "Example: ... blood_gas_analysis.yaml max_steps=60000 loss.lambda_rec=1.0"
        )
        sys.exit(1)

    yaml_path = sys.argv[1]
    overrides = sys.argv[2:]
    cfg = load_config(yaml_path, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
