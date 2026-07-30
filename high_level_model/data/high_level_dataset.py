import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms import v2 as T  # 使用 torchvision v2 的 transforms

# 设置日志
logging.basicConfig(level=logging.INFO)


def _get_ep_bounds(meta, ep_idx: int) -> tuple[int, int]:
    """Return (ep_start, ep_end) global frame indices for episode ep_idx.

    Supports lerobot ≥3.0 (meta.episodes HF Dataset with dataset_from/to_index).
    Raises AttributeError with a clear message if the format is unrecognized.
    """
    ep_info = meta.episodes[ep_idx]
    return int(ep_info["dataset_from_index"]), int(ep_info["dataset_to_index"])


def _tasks_df_to_index_map(tasks_df) -> dict[int, str]:
    """Convert lerobot 3.0 meta.tasks DataFrame to {task_index: task_text} dict."""
    return {int(row.task_index): task_text for task_text, row in tasks_df.iterrows()}


class HighLevelSequenceDataset(Dataset):
    """
        一个数据集类，它包装了 LeRobotDataset，用于为高级策略（High-Level Policy）提供数据。

        它从 LeRobotDataset 中采样一个历史图像序列，并关联一个未来的指令。

        支持两种模式：
        1. 单任务模式：仅返回图像序列和子任务指令
        2. 多任务模式：返回图像序列、子任务指令和任务级别指令
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
        初始化 HighLevelSequenceDataset.
        Args:
            lerobot_dataset (LeRobotDataset): 已经实例化的 LeRobotDataset 对象。
            camera_names (list[str]): 要使用的摄像头名称列表。
            history_len (int): 序列中包含的历史帧数。
            prediction_offset (int): 从当前帧到目标指令帧的偏移量。
            history_skip_frame (int): 在历史帧之间跳过的帧数。
            use_command_in_meta (bool): True → 使用 meta.tasks 中的任务文本；
                                        False → 使用 meta.subtasks 中的子任务文本。
            subset_episodes (list[int]): 只加载指定的 episode 列表。
            return_task_instruction (bool): 是否返回任务级别指令（用于多任务训练）。
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
                    f"摄像头 '{cam}' 不存在于数据集的元数据中。可用摄像头: {self.dset.meta.camera_keys}"
                )

        # ============== 构建 task_index -> task_text 映射 ==============
        # lerobot 3.0: meta.tasks 是 pandas DataFrame，index=task文本, column="task_index"
        self.task_index_to_text: dict[int, str] = {}
        tasks_df = getattr(self.dset.meta, 'tasks', None)
        if tasks_df is not None and len(tasks_df) > 0:
            self.task_index_to_text = _tasks_df_to_index_map(tasks_df)
            logging.info(f"从 meta.tasks 加载了 {len(self.task_index_to_text)} 个任务指令")

        # 构建 episode_idx -> task_instruction 的映射（用于多任务训练）
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

            start_local = self.history_span
            end_local = episode_len - self.prediction_offset
            for local_idx in range(start_local, end_local):
                global_frame_idx = ep_start + local_idx
                self.samples.append((episode_idx, global_frame_idx))

        if len(self.samples) == 0:
            raise ValueError("没有找到任何满足要求的样本！请检查 history/offset 设置或数据长度。")
        if dropped_count > 0:
            logging.warning(
                f"已过滤 {dropped_count} 个过短的片段。"
                f" (最小需 {min_required_len + 1} 帧)"
            )
        else:
            logging.info(f"所有 {len(target_episodes)} 个片段均符合长度要求。")

    def _build_episode_task_mapping(self):
        """构建 episode_idx -> task_instruction 的映射。

        lerobot 3.0: meta.episodes[ep_idx]["tasks"] 直接保存了该 episode 的任务文本列表。
        """
        logging.info("正在构建 episode -> task 映射...")
        try:
            total_episodes = self.dset.meta.total_episodes
            for ep_idx in range(total_episodes):
                ep_info = self.dset.meta.episodes[ep_idx]
                ep_tasks = ep_info.get("tasks", [])
                if isinstance(ep_tasks, (list, tuple)) and len(ep_tasks) > 0:
                    self.episode_to_task[ep_idx] = ep_tasks[0]
                elif isinstance(ep_tasks, str) and ep_tasks:
                    self.episode_to_task[ep_idx] = ep_tasks
                else:
                    self.episode_to_task[ep_idx] = ""
            logging.info(f"成功构建了 {len(self.episode_to_task)} 个 episode -> task 映射")
        except Exception as e:
            logging.error(f"构建 episode -> task 映射失败: {e}")
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

        # 构建图像序列
        image_sequence = []
        for frame_idx in selected_indices:
            frame_data = self.dset[frame_idx]
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(frame_data[cam_name])
            image_sequence.append(torch.stack(all_cam_images, dim=0))
        image_sequence = torch.stack(image_sequence, dim=0)

        # 获取子任务指令
        # lerobot 3.0: frame["task"] = 任务文本, frame["subtask"] = 子任务文本（若存在）
        target_frame_data = self.dset[target_frame_abs]
        if self.use_command_in_meta:
            command_gt = target_frame_data["task"]
        else:
            command_gt = target_frame_data.get("subtask", target_frame_data["task"])

        if self.return_task_instruction:
            task_instruction = self.get_task_instruction(episode_idx)
            return image_sequence, command_gt, episode_idx, task_instruction

        return image_sequence, command_gt, episode_idx


def load_hl_data_from_lerobot(
        repo_id: str,
        camera_names: list[str],
        batch_size_train: int,
        batch_size_val: int,
        history_len: int = 5,
        prediction_offset: int = 10,
        history_skip_frame: int = 1,
        train_ratio: float = 0.85,
        num_workers: int = 8,
        use_command_in_meta: bool = False,
        return_task_instruction: bool = True,
):
    """
    从 LeRobotDataset 加载数据并创建用于高级策略训练的 DataLoader。
    """
    print(f"从 LeRobot 数据集 '{repo_id}' 加载数据...")
    print(f"{history_len=}, {history_skip_frame=}, {prediction_offset=}")

    image_transforms = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset_lerobot = LeRobotDataset(
        repo_id=repo_id,
        image_transforms=image_transforms,
    )
    total_episodes = full_dataset_lerobot.meta.total_episodes
    all_indices = np.arange(total_episodes)
    np.random.seed(42)
    np.random.shuffle(all_indices)

    split_idx = int(total_episodes * train_ratio)
    train_episodes = all_indices[:split_idx].tolist()
    val_episodes = all_indices[split_idx:].tolist()
    print(f"数据划分: 总片段 {total_episodes} -> 训练 {len(train_episodes)} / 验证 {len(val_episodes)}")

    train_dataset = HighLevelSequenceDataset(
        lerobot_dataset=full_dataset_lerobot,
        camera_names=camera_names,
        history_len=history_len,
        prediction_offset=prediction_offset,
        subset_episodes=train_episodes,
        history_skip_frame=history_skip_frame,
        use_command_in_meta=use_command_in_meta,
        return_task_instruction=return_task_instruction,
    )
    val_dataset = HighLevelSequenceDataset(
        lerobot_dataset=full_dataset_lerobot,
        camera_names=camera_names,
        subset_episodes=val_episodes,
        history_len=history_len,
        prediction_offset=prediction_offset,
        history_skip_frame=history_skip_frame,
        use_command_in_meta=use_command_in_meta,
        return_task_instruction=return_task_instruction,
    )
    print(f"样本帧数统计: 训练集 {len(train_dataset)} 帧, 验证集 {len(val_dataset)} 帧")

    def collate_fn(batch):
        batch_len = len(batch[0])
        image_sequences = torch.stack([item[0] for item in batch], dim=0)
        command_gts = [item[1] for item in batch]
        if batch_len >= 3:
            episode_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
            if batch_len == 4:
                task_instructions = [item[3] for item in batch]
                return image_sequences, command_gts, episode_indices, task_instructions
            return image_sequences, command_gts, episode_indices
        return image_sequences, command_gts

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, full_dataset_lerobot


if __name__ == "__main__":
    REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
    CAMERA_NAMES = ["observation.images.top"]
    BATCH_SIZE_TRAIN = 8
    BATCH_SIZE_VAL = 8
    try:
        train_loader, val_loader, _ = load_hl_data_from_lerobot(
            repo_id=REPO_ID,
            camera_names=CAMERA_NAMES,
            batch_size_train=BATCH_SIZE_TRAIN,
            batch_size_val=BATCH_SIZE_VAL,
            history_len=5,
            prediction_offset=50,
            history_skip_frame=5,
        )

        print("\n正在从训练 DataLoader 中获取一个批次...")
        image_batch, command_batch, *_ = next(iter(train_loader))
        print(f"\n图像批次形状: {image_batch.shape}")
        print(f"\n指令批次 (大小: {len(command_batch)}):")
        for i, cmd in enumerate(command_batch[:5]):
            print(f"  - 样本 {i}: '{cmd}'")

    except Exception as e:
        print(f"\n在加载或处理数据时发生错误: {e}")
        raise
