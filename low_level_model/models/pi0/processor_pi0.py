"""PI0 Pre/Post-Processor Pipelines (System 1 — Generative Executor).

Provides the pre-processing and post-processing pipelines for the PI0 policy
used as System 1 (Generative Executor) in Robo-MLT.

Pre-processor pipeline (observation → model-ready batch)
---------------------------------------------------------
1. RenameObservationsProcessorStep  – remap observation keys if needed
2. AddBatchDimensionProcessorStep   – wrap single-step data in a batch of 1
3. Pi0TaskSubtaskConcatProcessor    – concatenate task + subtask, add trailing '\\n'
4. TokenizerProcessorStep           – tokenise task string (PaliGemma tokenizer)
5. DeviceProcessorStep              – move tensors to the target device
6. NormalizerProcessorStep          – normalise state / action features

Post-processor pipeline (model output → robot action)
------------------------------------------------------
1. UnnormalizerProcessorStep        – inverse-normalise action predictions
2. DeviceProcessorStep              – move tensors to CPU

Usage::

    pre, post = make_pi0_pre_post_processors(config, dataset_stats=ds.meta.stats)

    # Inference
    batch = pre(raw_obs)
    action_chunk = policy.predict_action_chunk(batch)
    action_chunk = post(action_chunk)
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
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from low_level_model.models.pi0.configuration_pi0 import PI0Config


class Pi0TaskSubtaskConcatProcessor(ComplementaryDataProcessorStep):
    """Select the prompt text under 'task' and ensure a trailing newline.

    PaliGemma's tokenizer expects the text prompt to end with '\\n'.
    When a 'subtask' field is present it is used as the prompt (the high-level
    'task' is dropped); otherwise the 'task' is used. Result is stored under 'task'.
    """

    @staticmethod
    def _merge_text(task: str, subtask: str) -> str:
        # Feed the model ONLY the subtask (current atomic skill) when present;
        # fall back to the high-level task when no subtask is available.
        merged = subtask if subtask and subtask.strip() else task
        return merged if merged.endswith("\n") else f"{merged}\n"

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


class Pi0NewLineProcessor(ComplementaryDataProcessorStep):
    """Ensure the task description ends with a newline character.

    Simpler alternative to Pi0TaskSubtaskConcatProcessor when only task
    (no subtask) is present.
    """

    def complementary_data(self, complementary_data: dict) -> dict:
        if "task" not in complementary_data:
            return complementary_data
        task = complementary_data["task"]
        if task is None:
            return complementary_data
        new_data = dict(complementary_data)
        if isinstance(task, str):
            new_data["task"] = task if task.endswith("\n") else f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            new_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        return new_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi0_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre-processor and post-processor pipelines for the PI0 policy.

    Args:
        config: PI0 policy configuration.
        dataset_stats: Per-feature normalisation statistics (e.g. ``ds.meta.stats``).
                       Pass None when statistics will be loaded from a checkpoint.

    Returns:
        ``(preprocessor, postprocessor)`` pipeline tuple.
    """
    norm_map = {FeatureType(ft): mode for ft, mode in config.normalization_mapping.items()}
    all_features = {**config.input_features, **config.output_features}

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        Pi0TaskSubtaskConcatProcessor(),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        ),
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


__all__ = [
    "Pi0TaskSubtaskConcatProcessor",
    "Pi0NewLineProcessor",
    "make_pi0_pre_post_processors",
]
