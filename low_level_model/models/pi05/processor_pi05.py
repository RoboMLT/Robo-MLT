#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from low_level_model.models.pi05.configuration_pi05 import PI05FullConfig
from low_level_model.models.pi05.modeling_pi05 import pad_vector

from lerobot.processor import (
    ActionTokenizerProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.type import (EnvTransition, TransitionKey)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK,
    OBS_LANGUAGE_USER_PROMPT_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """
    max_state_dim: int = 32
    user_prompt_key: str = "task"
    command_key: str = "subtask"
    advantage_scaling: float = 1.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")

        # DEBUG: Check complementary data
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        user_prompts = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.user_prompt_key)
        if user_prompts is None:
            raise ValueError("No user prompts found in complementary data")
        # Normalize a single bare string to a 1-element list. At inference the batch carries a
        # scalar `task` (and `subtask`) string; without this, `enumerate(...)` below iterates the
        # string's CHARACTERS, producing one bogus per-character prompt each.
        if isinstance(user_prompts, str):
            user_prompts = [user_prompts]

        # At inference the model generates the subtask itself (autoregressive decode inside
        # predict_action_chunk), so no ground-truth subtask is present. Only require it at
        # training time, where lerobot always injects `subtask` from meta/subtasks.parquet.
        commands = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.command_key)
        if isinstance(commands, str):
            commands = [commands]

        #TODO pi0.6
        # # Check for advantage
        # advantages = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("advantage")
        # if advantages is not None and isinstance(advantages, torch.Tensor):
        #     advantages = advantages.cpu().float().numpy().flatten()

        # TODO: check if this necessary
        state = deepcopy(state)


        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)
        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().float().numpy()

        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        critic_prompts = []
        for i, user_prompt in enumerate(user_prompts):
            cleaned_text = user_prompt.strip().replace("_", " ").replace("\n", " ")
            cleaned_text = cleaned_text.lower()   # all lowercase # NOTE: added by (jadechoghari)

            state_str = " ".join(map(str, discretized_states[i]))

            # advantage_str = ""
            # if advantages is not None:
            #     # Scale and bin advantage
            #     adv = advantages[i] / self.advantage_scaling
            #     adv = np.tanh(adv)
            #
            #     bins = np.array([-1.0, 0.35, 1.0])
            #     # Clip to range
            #     adv = np.clip(adv, -1.0, 1.0)
            #     # digitize
            #     adv_bin = np.digitize(adv, bins) - 1  # 0 to 3
            #     # Clamp to 0-2 (3 bins)
            #     adv_bin = max(0, min(1, adv_bin))
            #     # Map bin index to string label
            #     labels = ["negative", "positive"]
            #     adv_label = labels[adv_bin]

            # Format: "Advantage: <label>"
            full_prompt = f"Task: {cleaned_text}, State: {state_str}\n"
            full_prompts.append(full_prompt)
            critic_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
            critic_prompts.append(critic_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.user_prompt_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA]["critic_prompt"] = critic_prompts

        # process commands (training only; skipped at inference where subtask is absent)
        if commands is not None:
            full_commands = []
            for i, command in enumerate(commands):
                cleaned_text = command.strip().replace("_", " ").replace("\n", " ")
                cleaned_text = cleaned_text.lower()   # all lowercase # NOTE: added by (jadechoghari)
                full_command = f"Subtask: {cleaned_text};\n"
                full_commands.append(full_command)

            transition[TransitionKey.COMPLEMENTARY_DATA][self.command_key] = full_commands
        # note: action tokens will be processed in the ActionTokenizerProcessorStep
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi05_tokenizer_processor")
@dataclass
class Pi05TokenizerProcessorStep(TokenizerProcessorStep):
    """PaliGemma tokenizer step specialised for π0.5.

    Two differences from the stock ``TokenizerProcessorStep``:

    1. The tokenized task prompt (``Task: ..., State: <bins>\\n``) is also aliased into the
       inference-time keys ``OBS_LANGUAGE_USER_PROMPT_TOKENS`` / ``_ATTENTION_MASK`` that
       ``PI05FullPolicy.predict_action_chunk`` reads. Training reads ``OBS_LANGUAGE_TOKENS``;
       both hold the identical prompt, so a single tokenization serves both paths.
    2. Subtask targets are tokenized with a dedicated tokenizer that appends EOS
       (``add_eos_token=True``) and a shorter ``subtask_max_length``. Without the EOS the
       autoregressive ``generate_subtask_tokens`` decode (which stops on ``eos_token_id``)
       would never learn to terminate and would emit ``max_decoding_steps`` of garbage tail.
    """

    subtask_max_length: int = 32
    subtask_add_eos: bool = True

    # Internal subtask tokenizer instance (not part of the config)
    subtask_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # Build a separate tokenizer that appends EOS so the model learns to stop decoding.
        # BOS is kept (default) to match the BOS-seeded subtask segment at inference.
        if self.subtask_add_eos and self.tokenizer_name is not None:
            self.subtask_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, add_eos_token=True
            )
        else:
            self.subtask_tokenizer = self.input_tokenizer

    def _tokenize_subtask(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        return self.subtask_tokenizer(
            text,
            max_length=self.subtask_max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def observation(self, observation):
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        tokenized_prompt = self._tokenize_text(task)
        target_device = self._detect_device(self.transition)
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        new_observation = dict(observation)
        input_ids = tokenized_prompt["input_ids"]
        attn_mask = tokenized_prompt["attention_mask"].to(dtype=torch.bool)
        new_observation[OBS_LANGUAGE_TOKENS] = input_ids
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = attn_mask
        # Alias into the inference-time user-prompt keys (predict_action_chunk reads these);
        # harmless duplicates during training, which reads OBS_LANGUAGE_TOKENS instead.
        new_observation[OBS_LANGUAGE_USER_PROMPT_TOKENS] = input_ids
        new_observation[OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK] = attn_mask

        # Subtask (training only): tokenize with EOS appended.
        subtask = self.get_subtask(self.transition)
        if subtask is not None:
            tokenized_subtask = self._tokenize_subtask(subtask)
            if target_device is not None:
                tokenized_subtask = {
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_subtask.items()
                }
            new_observation[OBS_LANGUAGE_SUBTASK_TOKENS] = tokenized_subtask["input_ids"]
            new_observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = tokenized_subtask[
                "attention_mask"
            ].to(dtype=torch.bool)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["subtask_max_length"] = self.subtask_max_length
        config["subtask_add_eos"] = self.subtask_add_eos
        return config


def make_pi05_full_pre_post_processors(
    config: PI05FullConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
)-> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
     Constructs pre-processor and post-processor pipelines for the PI0 policy.

     The pre-processing pipeline prepares input data for the model by:
     1. Renaming features to match pretrained configurations.
     2. Normalizing input and output features based on dataset statistics.
     3. Adding a batch dimension.
     4. Appending a newline character to the task description for tokenizer compatibility.
     5. Tokenizing the text prompt using the PaliGemma tokenizer.
     6. Moving all data to the specified device.

     The post-processing pipeline handles the model's output by:
     1. Moving data to the CPU.
     2. Unnormalizing the output features to their original scale.

     Args:
         config: The configuration object for the PI0 policy.
         dataset_stats: A dictionary of statistics for normalization.
         preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
         postprocessor_kwargs: Additional arguments for the post-processor pipeline.

     Returns:
         A tuple containing the configured pre-processor and post-processor pipelines.
    """
    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            **(preprocessor_overrides.get("pi05_full_prepare_state_tokenizer_processor_step",
                                          {}) if preprocessor_overrides else {})
        ),
        Pi05TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            subtask_max_length=config.subtask_max_length,
            padding_side="right",
            padding="max_length",
        ),
        ActionTokenizerProcessorStep(
            action_tokenizer_name=config.action_tokenizer_name,
            max_action_tokens=config.max_action_tokens,
            fast_skip_tokens=config.fast_skip_tokens,
            paligemma_tokenizer_name=config.text_tokenizer_name,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


