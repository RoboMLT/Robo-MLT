"""Temporal-delay augmentation datasets for System 1 training.

These datasets extend ``LeRobotDataset`` to implement Robo-MLT's asynchronous
training strategy: during training the action chunk is shifted forward in time
by a random offset in ``[0, max_delay_steps]``.  This teaches the policy to
predict where the robot *will* be rather than where it currently is, so that at
deployment System 1 can begin predicting the next chunk before the current one
finishes (see ``System1AsyncStreamer``).

State handling under an offset:
    - offset == 0 : use the original ``observation.state``.
    - offset  > 0 : use the previous action ``a_{idx+offset-1}`` as the state
      (matching the async-offset semantics used at inference).

Cross-subtask action masking (``mask_cross_subtask_actions``):
    A dataset frame may sit near a subtask boundary, so a chunk of
    ``chunk_size`` actions can straddle two subtasks (e.g. the first 25 steps
    belong to the current skill, the last 25 to the next one). Supervising the
    whole chunk teaches the policy to predict the *next* subtask's actions while
    still conditioned on the *current* subtask's language prompt — chunks blur
    together at deployment. When enabled, the decode-free integer
    ``subtask_index`` column is read once and every action step at or after the
    first subtask transition (relative to the observation frame ``idx``, which is
    also what ``SubtaskInjectingDataset`` reads for the language prompt) is
    flagged in ``action_is_pad`` so it is excluded from the loss. The policy is
    thus only supervised on actions that belong to the skill it is being told to
    execute.

Two variants are provided:

``DelayAugmentedDataset``
    Samples a single random offset per item.  Drop-in replacement for the plain
    dataset.

``SharedObservationDataset``
    Returns *all* valid offsets ``[0, max_offset]`` for one observation, so the
    expensive image/language prefix is embedded once and reused across offsets
    (≈ ``(max_delay_steps + 1)x`` fewer prefix forwards).  Use together with
    :func:`shared_observation_collate_fn` and ``PI0Policy.forward_shared_observation``.
"""

import logging
from pathlib import Path
from typing import Callable
import random

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


class DelayAugmentedDataset(LeRobotDataset):
    """LeRobotDataset with random temporal-delay augmentation.

    Example (chunk_size=50, max_delay_steps=12):
        - offset 0 : actions [t, t+1, ..., t+49], state = obs.state at t
        - offset 5 : actions [t+5, ..., t+54], state = action at t+4
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        max_delay_steps: int = 0,
        mask_cross_subtask_actions: bool = False,
        subtask_index_column: str = "subtask_index",
    ):
        self.max_delay_steps = max_delay_steps
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        # Offset chosen in _get_query_indices, consumed in __getitem__.
        self._last_offset: int = 0
        # Decode-free per-frame subtask labels for cross-subtask action masking.
        self.mask_cross_subtask_actions = mask_cross_subtask_actions
        self.subtask_index_column = subtask_index_column
        self._subtask_index_col: list[int] | None = None
        if mask_cross_subtask_actions:
            self._subtask_index_col = self._load_subtask_index_column(subtask_index_column)

    def _load_subtask_index_column(self, column: str) -> list[int] | None:
        """Read the integer subtask-index column once (no image/video decode)."""
        hf = getattr(self, "hf_dataset", None)
        if hf is None or column not in getattr(hf, "column_names", []):
            logger.warning(
                "mask_cross_subtask_actions=True but column '%s' is absent from hf_dataset; "
                "cross-subtask masking is disabled.", column,
            )
            return None
        logger.info("Cross-subtask action masking enabled (column '%s', decode-free).", column)
        return [int(x) for x in hf[column]]

    def _cross_subtask_action_pad(
        self, idx: int, action_query_indices: list[int]
    ) -> torch.BoolTensor | None:
        """Pad mask truncating the action chunk at its first subtask transition.

        The anchor subtask is read at the observation frame ``idx`` (matching the
        language prompt injected by ``SubtaskInjectingDataset``). Every step at or
        after the first action frame whose subtask differs from the anchor is
        flagged ``True`` so it is dropped from the loss.
        """
        col = self._subtask_index_col
        if col is None:
            return None
        anchor = col[idx]
        mask, crossed = [], False
        for ai in action_query_indices:
            crossed = crossed or (col[ai] != anchor)
            mask.append(crossed)
        return torch.BoolTensor(mask)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """Shift all action delta indices by a random offset, clamped to the episode."""
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Largest offset that keeps the last action in-episode.
        max_delta = self.delta_indices["action"][-1]
        max_offset = min(self.max_delay_steps, max(0, ep_end - 1 - (idx + max_delta)))
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
        self._last_offset = offset

        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.BoolTensor] = {}
        for key, delta_idx in self.delta_indices.items():
            query_indices[key] = [
                max(ep_start, min(ep_end - 1, idx + delta + offset)) for delta in delta_idx
            ]
            padding[f"{key}_is_pad"] = torch.BoolTensor(
                [(idx + delta + offset < ep_start) | (idx + delta + offset >= ep_end) for delta in delta_idx]
            )

        cross = self._cross_subtask_action_pad(idx, query_indices["action"])
        if cross is not None:
            padding["action_is_pad"] = padding["action_is_pad"] | cross
        return query_indices, padding

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)

        offset = getattr(self, "_last_offset", 0)
        if offset <= 0:
            return item

        ep_idx = item["episode_index"].item() if "episode_index" in item else None
        if ep_idx is None:
            raise ValueError("episode_index not found in item")

        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Previous action becomes the state when the chunk is delayed.
        prev_idx = max(ep_start, min(ep_end - 1, idx + offset - 1))
        obs_state = item["observation.state"]
        prev_action = self.hf_dataset[prev_idx]["action"]

        if obs_state.dim() != 1 or prev_action.dim() != 1:
            raise ValueError("Only 1D state/action are supported for delay augmentation.")
        if obs_state.shape[0] != prev_action.shape[0]:
            raise ValueError(
                "state_dim != action_dim is unsupported when applying an async offset "
                "to observation.state."
            )

        item["observation.state"] = prev_action
        return item


class SharedObservationDataset(DelayAugmentedDataset):
    """Return every valid offset for one observation (shared-observation training).

    Each item carries a single observation plus per-offset states/actions, so the
    prefix (images + language) is embedded once and reused across all offsets via
    ``PI0Policy.forward_shared_observation``.
    """

    def _get_query_indices_for_offset(
        self, idx: int, ep_idx: int, offset: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.BoolTensor]]:
        """Action query indices for a specific (non-random) offset."""
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.BoolTensor] = {}
        for key, delta_idx in self.delta_indices.items():
            query_indices[key] = [
                max(ep_start, min(ep_end - 1, idx + delta + offset)) for delta in delta_idx
            ]
            padding[f"{key}_is_pad"] = torch.BoolTensor(
                [(idx + delta + offset < ep_start) | (idx + delta + offset >= ep_end) for delta in delta_idx]
            )

        cross = self._cross_subtask_action_pad(idx, query_indices["action"])
        if cross is not None:
            padding["action_is_pad"] = padding["action_is_pad"] | cross
        return query_indices, padding

    def __getitem__(self, idx) -> dict:
        ep_idx = self.hf_dataset[idx]["episode_index"].item()
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        max_delta = self.delta_indices["action"][-1]
        max_offset = min(self.max_delay_steps, max(0, ep_end - 1 - (idx + max_delta)))
        num_offsets = max_offset + 1

        # Base item with offset 0: bypass DelayAugmentedDataset's random offset by
        # calling the grandparent __getitem__ directly.
        self._last_offset = 0
        base_item = super(DelayAugmentedDataset, self).__getitem__(idx)

        result: dict = {}
        for key in base_item:
            if (
                key.startswith("observation.images.")
                or key in ("task", "episode_index", "subtask")
            ):
                result[key] = base_item[key]

        states, actions, action_is_pads = [], [], []
        for offset in range(num_offsets):
            if offset == 0:
                state = base_item["observation.state"]
            else:
                prev_idx = max(ep_start, min(ep_end - 1, idx + offset - 1))
                prev_action = self.hf_dataset[prev_idx]["action"]
                obs_state = base_item["observation.state"]
                if obs_state.dim() != 1 or prev_action.dim() != 1:
                    raise ValueError("Only 1D state/action are supported for delay augmentation.")
                if obs_state.shape[0] != prev_action.shape[0]:
                    raise ValueError(
                        "state_dim != action_dim is unsupported when applying an async offset."
                    )
                state = prev_action
            states.append(state)

            query_indices, padding = self._get_query_indices_for_offset(idx, ep_idx, offset)
            action_list = [self.hf_dataset[ai]["action"] for ai in query_indices["action"]]
            actions.append(torch.stack(action_list, dim=0))            # [chunk, action_dim]
            action_is_pads.append(padding["action_is_pad"])             # [chunk]

        result["observation.state"] = torch.stack(states, dim=0)        # [num_offsets, state_dim]
        result["action"] = torch.stack(actions, dim=0)                  # [num_offsets, chunk, action_dim]
        result["action_is_pad"] = torch.stack(action_is_pads, dim=0)    # [num_offsets, chunk]
        result["num_offsets"] = num_offsets
        return result


def shared_observation_collate_fn(batch: list[dict]) -> dict:
    """Collate fn for :class:`SharedObservationDataset`.

    Pads the per-offset tensors to ``max_offsets`` within the batch and emits an
    ``offset_mask`` marking which offsets are real.

    Returns a batch with:
        - shared observation keys batched normally,
        - ``observation.state`` : [B, max_offsets, state_dim],
        - ``action``            : [B, max_offsets, chunk, action_dim],
        - ``action_is_pad``     : [B, max_offsets, chunk] (bool),
        - ``offset_mask``       : [B, max_offsets] (bool),
        - ``max_offsets``       : int.
    """
    max_offsets = max(item["num_offsets"] for item in batch)

    per_offset_keys = {"observation.state", "action", "action_is_pad"}
    shared_keys = [k for k in batch[0] if k not in per_offset_keys and k != "num_offsets"]

    result: dict = {}
    result.update(default_collate([{k: item[k] for k in shared_keys} for item in batch]))

    batch_size = len(batch)
    device = batch[0]["observation.state"].device
    state_shape = batch[0]["observation.state"].shape[1:]   # [state_dim]
    action_shape = batch[0]["action"].shape[1:]             # [chunk, action_dim]

    padded_states = torch.zeros(batch_size, max_offsets, *state_shape, device=device)
    padded_actions = torch.zeros(batch_size, max_offsets, *action_shape, device=device)
    padded_action_is_pad = torch.ones(
        batch_size, max_offsets, action_shape[0], dtype=torch.bool, device=device
    )
    offset_mask = torch.zeros(batch_size, max_offsets, dtype=torch.bool, device=device)

    for i, item in enumerate(batch):
        n = item["num_offsets"]
        padded_states[i, :n] = item["observation.state"]
        padded_actions[i, :n] = item["action"]
        padded_action_is_pad[i, :n] = item["action_is_pad"]
        offset_mask[i, :n] = True

    result["observation.state"] = padded_states
    result["action"] = padded_actions
    result["action_is_pad"] = padded_action_is_pad
    result["offset_mask"] = offset_mask
    result["max_offsets"] = max_offsets
    return result


__all__ = [
    "DelayAugmentedDataset",
    "SharedObservationDataset",
    "shared_observation_collate_fn",
]
