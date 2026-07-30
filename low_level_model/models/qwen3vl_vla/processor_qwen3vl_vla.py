"""Qwen3VL-VLA Pre/Post-Processor Pipelines (System 1 — Generative Executor).

Pre-processor pipeline (observation → model-ready batch)
---------------------------------------------------------
1. RenameObservationsProcessorStep  – remap observation keys if needed
2. AddBatchDimensionProcessorStep   – wrap single-step data in a batch of 1
3. QwenTaskSubtaskConcatProcessor   – merge task + subtask into the `task` string
4. DeviceProcessorStep              – move tensors to the target device
5. NormalizerProcessorStep          – normalise state / action (images are IDENTITY)

There is **no** tokenizer step: unlike PI0/SmolVLA, the Qwen backbone tokenises
raw PIL images + text itself via ``processor.apply_chat_template`` inside the
model, so the language must reach the policy as the plain ``task`` string (not
pre-tokenised ids).

Post-processor pipeline (model output → robot action)
------------------------------------------------------
1. UnnormalizerProcessorStep        – inverse-normalise action predictions
2. DeviceProcessorStep              – move tensors to CPU
"""

from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from low_level_model.models.qwen3vl_vla.configuration_qwen3vl_vla import Qwen3VLVLAConfig


class QwenTaskSubtaskConcatProcessor(ComplementaryDataProcessorStep):
    """Merge the 'task' and optional 'subtask' fields into a single 'task' string.

    Mirrors the Robo-MLT deployment prompt "task: <high> subtask: <atomic>" that
    System 2 feeds at runtime.  Unlike PaliGemma, Qwen's chat template needs no
    trailing newline, so none is added.
    """

    @staticmethod
    def _merge_text(task: str, subtask: str) -> str:
        if subtask and subtask.strip():
            return f"task: {task} subtask: {subtask}".strip()
        return task

    def complementary_data(self, complementary_data: dict) -> dict:
        if "task" not in complementary_data:
            return complementary_data
        task = complementary_data.get("task")
        subtask = complementary_data.get("subtask")
        if task is None:
            return complementary_data
        new_data = dict(complementary_data)
        if isinstance(task, str):
            s = subtask if isinstance(subtask, str) else ""
            new_data["task"] = self._merge_text(task, s)
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            if isinstance(subtask, list) and len(subtask) == len(task) and all(isinstance(s, str) for s in subtask):
                new_data["task"] = [self._merge_text(t, s) for t, s in zip(task, subtask)]
            elif isinstance(subtask, str):
                new_data["task"] = [self._merge_text(t, subtask) for t in task]
            else:
                new_data["task"] = [self._merge_text(t, "") for t in task]
        return new_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_qwen3vl_vla_pre_post_processors(
    config: Qwen3VLVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre-processor and post-processor pipelines for the Qwen3VL-VLA policy."""
    norm_map = {FeatureType(ft): mode for ft, mode in config.normalization_mapping.items()}
    all_features = {**config.input_features, **config.output_features}

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        QwenTaskSubtaskConcatProcessor(),
        DeviceProcessorStep(device=config.device or "cpu"),
        NormalizerProcessorStep(features=all_features, norm_map=norm_map, stats=dataset_stats),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(features=config.output_features, norm_map=norm_map, stats=dataset_stats),
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


__all__ = ["QwenTaskSubtaskConcatProcessor", "make_qwen3vl_vla_pre_post_processors"]
