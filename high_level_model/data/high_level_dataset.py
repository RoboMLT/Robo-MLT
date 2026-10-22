"""History-window sampling over a LeRobot v3 dataset (base class of the System-2 dataset)."""

import logging

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__ = ["HighLevelSequenceDataset"]


def _get_ep_bounds(meta, ep_idx: int) -> tuple[int, int]:
    """Return (ep_start, ep_end) global frame indices for episode ep_idx (LeRobot >= 3.0)."""
    ep_info = meta.episodes[ep_idx]
    return int(ep_info["dataset_from_index"]), int(ep_info["dataset_to_index"])


def _tasks_df_to_index_map(tasks_df) -> dict[int, str]:
    """Convert the LeRobot 3.0 ``meta.tasks`` DataFrame to {task_index: task_text}."""
    return {int(row.task_index): task_text for task_text, row in tasks_df.iterrows()}


class HighLevelSequenceDataset(Dataset):
    """Wrap a LeRobotDataset and sample a history of multiview frames per anchor frame.

    Each sample is the window of ``history_len`` frames, ``history_skip_frame`` apart, that ends at
    the anchor frame, together with the (sub)task instruction ``prediction_offset`` frames later.
    With ``return_task_instruction`` the episode-level task text is returned as well.
    """

    def __init__(
            self,
            lerobot_dataset: LeRobotDataset,
            camera_names: list[str],
            history_len: int = 5,
            prediction_offset: int = 20,
            history_skip_frame: int = 10,
            use_command_in_meta: bool = True,
            subset_episodes: list[int] = None,
            return_task_instruction: bool = False,
    ):
        """
        Args:
            lerobot_dataset: an instantiated LeRobotDataset.
            camera_names: observation image keys to stack, in order.
            history_len: number of frames in the history window.
            prediction_offset: offset from the anchor frame to the frame whose instruction is used.
            history_skip_frame: spacing between consecutive history frames.
            use_command_in_meta: True -> task text from ``meta.tasks``;
                False -> subtask text from ``meta.subtasks``.
            subset_episodes: only sample from these episodes.
            return_task_instruction: also return the episode-level task text.
        """
        super().__init__()
        self.dset = lerobot_dataset
        self.camera_names = camera_names
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.use_command_in_meta = use_command_in_meta
        self.return_task_instruction = return_task_instruction

        for cam in self.camera_names:
            if cam not in self.dset.meta.camera_keys:
                raise ValueError(
                    f"Camera '{cam}' is not in the dataset metadata. "
                    f"Available cameras: {self.dset.meta.camera_keys}"
                )

        # LeRobot 3.0: meta.tasks is a DataFrame with index = task text, column "task_index".
        self.task_index_to_text: dict[int, str] = {}
        tasks_df = getattr(self.dset.meta, "tasks", None)
        if tasks_df is not None and len(tasks_df) > 0:
            self.task_index_to_text = _tasks_df_to_index_map(tasks_df)
            logger.info("Loaded %d task instructions from meta.tasks", len(self.task_index_to_text))

        self.episode_to_task: dict[int, str] = {}
        if return_task_instruction:
            self._build_episode_task_mapping()

        self.history_span = self.history_len * self.history_skip_frame
        min_required_len = self.history_span + self.prediction_offset
        if subset_episodes is not None:
            target_episodes = subset_episodes
        else:
            target_episodes = (
                self.dset.episodes
                if self.dset.episodes is not None
                else range(self.dset.meta.total_episodes)
            )

        self.samples: list[tuple[int, int]] = []
        dropped_count = 0
        for episode_idx in target_episodes:
            ep_start, ep_end = _get_ep_bounds(self.dset.meta, episode_idx)
            episode_len = ep_end - ep_start
            if episode_len <= min_required_len:
                dropped_count += 1
                continue
            for local_idx in range(self.history_span, episode_len - self.prediction_offset):
                self.samples.append((episode_idx, ep_start + local_idx))

        if len(self.samples) == 0:
            raise ValueError("No sample satisfies the history/offset settings; check them against "
                             "the episode lengths.")
        if dropped_count > 0:
            logger.warning("Skipped %d episodes shorter than %d frames.",
                           dropped_count, min_required_len + 1)
        else:
            logger.info("All %d episodes are long enough.", len(target_episodes))

    def _build_episode_task_mapping(self):
        """episode_idx -> task text (LeRobot 3.0 stores the list in meta.episodes[ep]["tasks"])."""
        try:
            for ep_idx in range(self.dset.meta.total_episodes):
                ep_tasks = self.dset.meta.episodes[ep_idx].get("tasks", [])
                if isinstance(ep_tasks, (list, tuple)) and len(ep_tasks) > 0:
                    self.episode_to_task[ep_idx] = ep_tasks[0]
                elif isinstance(ep_tasks, str) and ep_tasks:
                    self.episode_to_task[ep_idx] = ep_tasks
                else:
                    self.episode_to_task[ep_idx] = ""
            logger.info("Built the episode -> task map for %d episodes", len(self.episode_to_task))
        except Exception as e:  # noqa: BLE001 - a missing column must not abort training
            logger.error("Building the episode -> task map failed: %s", e)
            for ep_idx in range(self.dset.meta.total_episodes):
                self.episode_to_task[ep_idx] = ""

    def get_task_instruction(self, episode_idx: int) -> str:
        return self.episode_to_task.get(episode_idx, "")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_idx, curr_frame_abs = self.samples[index]
        target_frame_abs = curr_frame_abs + self.prediction_offset
        start_frame_abs = curr_frame_abs - self.history_span
        raw_indices = list(range(start_frame_abs, curr_frame_abs + 1, self.history_skip_frame))
        selected_indices = raw_indices[-self.history_len:]

        image_sequence = []
        for frame_idx in selected_indices:
            frame_data = self.dset[frame_idx]
            image_sequence.append(torch.stack([frame_data[c] for c in self.camera_names], dim=0))
        image_sequence = torch.stack(image_sequence, dim=0)

        # LeRobot 3.0: frame["task"] is the task text, frame["subtask"] the subtask text if present.
        target_frame_data = self.dset[target_frame_abs]
        if self.use_command_in_meta:
            command_gt = target_frame_data["task"]
        else:
            command_gt = target_frame_data.get("subtask", target_frame_data["task"])

        if self.return_task_instruction:
            return image_sequence, command_gt, episode_idx, self.get_task_instruction(episode_idx)
        return image_sequence, command_gt, episode_idx
