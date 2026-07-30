"""PI0 Model Implementation (System 1 — Generative Executor).

This module implements the PI0 (π0) Vision-Language-Action model for
robot control. PI0 uses flow matching to generate action sequences
conditioned on images and language instructions.

Architecture:
    PI0Policy (wrapper)
    └── PI0Model (core model)
        ├── PaliGemmaForConditionalGeneration (vision-language backbone)
        ├── GemmaForCausalLM (action expert, no adaRMS)
        ├── PI0PrefixEmbedder (image + language embeddings)
        ├── PI0SuffixEmbedder (state + noisy action + time embeddings)
        └── PI0ModelLayer[] (shared transformer layers, one per backbone layer)

Flow Matching training objective:
    x_t = t * noise + (1-t) * actions        (linear interpolation)
    u_t = noise - actions                     (target velocity)
    loss = MSE(v_θ(x_t, t, context), u_t)   (velocity prediction loss)

Inference (Euler integration, t: 1 → 0):
    x_{t+dt} = x_t + dt * v_θ(x_t, t, context)

This is the System 1 (Generative Executor) in the Robo-MLT dual-system
framework described in the accompanying paper.
"""

import builtins
import contextlib
import logging
import math
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from lerobot.configs.policies import PreTrainedConfig, T
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
)

from low_level_model.models.pi0.configuration_pi0 import PI0Config
from low_level_model.utils.pi0_utils import (
    create_sinusoidal_pos_embedding,
    pad_vector,
    build_attention_mask_and_position_ids,
    build_shared_obs_attention_mask_and_position_ids,
    resize_with_pad,
)
from low_level_model.models.layers.attention import Attention
from low_level_model.models.layers.linear import QKVLinear, MergedColumnLinear
from low_level_model.models.layers.rope import RotaryEmbedding

logger = logging.getLogger(__name__)


class PI0PrefixEmbedder(nn.Module):
    """Embed images and language tokens into the prefix (conditioning) sequence.

    Images are processed by SigLIP; language tokens are embedded and scaled
    by sqrt(hidden_dim) following the PaliGemma convention.
    """

    def __init__(self, config: PI0Config, vlm: PaliGemmaForConditionalGeneration):
        super().__init__()
        self.config = config
        self.img_embedder = vlm.model.get_image_features
        self.lang_embedder = vlm.language_model.embed_tokens

    def forward(self, images, img_masks, tokens, masks):
        """Embed images and language into prefix sequence.

        Args:
            images: List of image tensors [B, C, H, W].
            img_masks: List of boolean validity masks [B].
            tokens: Language token IDs [B, L_text].
            masks: Language attention masks [B, L_text].

        Returns:
            embs: Concatenated embeddings [B, L_prefix, D].
            pad_masks: Padding masks [B, L_prefix].
            att_masks: Attention pattern masks [B, L_prefix].
        """
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            if img.dtype != torch.float32:
                img = img.to(torch.float32)
            img_emb = self.img_embedder(img)
            bsz, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsz, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.lang_embedder(tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        bsz = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsz, len(att_masks))

        return embs, pad_masks, att_masks


class PI0SuffixEmbedder(nn.Module):
    """Embed state, noisy actions, and timestep for flow matching.

    The suffix carries all inputs that change per denoising step:
    a state token followed by action tokens (each fused with the timestep).
    """

    def __init__(self, config: PI0Config):
        super().__init__()
        self.config = config
        width = config.action_expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.state_proj = nn.Linear(config.max_state_dim, width)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

    def forward(self, state, noisy_actions, time):
        """Embed state and noisy actions with time conditioning.

        Args:
            state: Robot state [B, state_dim].
            noisy_actions: Noisy action sequence x_t [B, T, action_dim].
            time: Flow matching timestep. Either a per-sample scalar ``[B]``
                (baseline pi0, broadcast across all action tokens) or a per-token
                timestep ``[B, T]`` (TTRTC: the clean action prefix uses ``t=0``
                while the noisy postfix uses the sampled ``t``).

        Returns:
            embs: Suffix embeddings [B, 1+T, D].
            pad_masks: Padding masks [B, 1+T].
            att_masks: Attention masks [B, 1+T].
            adarms_cond: None (PI0 doesn't use adaRMS conditioning).
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Cast all inputs to the weight dtype (bfloat16 during training) so every
        # F.linear call sees matching dtypes regardless of how inputs arrive.
        w_dtype = self.state_proj.weight.dtype
        state = state.to(dtype=w_dtype)
        noisy_actions = noisy_actions.to(dtype=w_dtype)

        # State token (first suffix token)
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks.append(1)  # boundary: prefix doesn't attend to suffix

        # Time embedding (sinusoidal) — cast to w_dtype to match action embeddings
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.action_expert_config.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=time.device,
        ).to(dtype=w_dtype)


        action_emb = self.action_in_proj(noisy_actions)
        if time_emb.ndim == 2:
            time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
        else:
            time_emb_expanded = time_emb
        action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
        action_time_emb = F.silu(self.action_time_mlp_in(action_time_emb))
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        embs.append(action_time_emb)

        bsize_t, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize_t, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        # State token cannot attend to action tokens (causal boundary)
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        adarms_cond = None  # PI0 has no adaRMS conditioning
        return embs, pad_masks, att_masks, adarms_cond


class PI0Attention(nn.Module):
    """Joint cross-backbone attention over VLM and action-expert hidden states."""

    def __init__(
        self,
        config: PI0Config,
        vlm_attention: nn.Module,
        action_expert_attention: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.vlm_attention = vlm_attention
        self.action_expert_attention = action_expert_attention
        text_cfg = config.vlm_config.text_config

        self.rotary_emb = RotaryEmbedding(
            head_size=text_cfg.head_dim,
            rotary_dim=text_cfg.head_dim,
            max_position_embeddings=text_cfg.max_position_embeddings,
            base=text_cfg.rope_theta,
        )
        self.attn = Attention(scale=vlm_attention.scaling)
        self.num_heads = text_cfg.num_attention_heads
        self.head_dim = text_cfg.head_dim

    def forward(self, hidden_states, attention_mask, position_ids, conds, use_cache: bool = False):
        attns = [self.vlm_attention, self.action_expert_attention]

        q_states, k_states, v_states = [], [], []
        for attn, hs in zip(attns, hidden_states):
            if hs is None or hs.shape[1] == 0:
                continue
            if hasattr(attn, "qkv_proj"):
                q, k, v = attn.qkv_proj(hs)
            else:
                bsz, seqlen, _ = hs.shape
                q = attn.q_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
                k = attn.k_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
                v = attn.v_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
            q_states.append(q)
            k_states.append(k)
            v_states.append(v)

        q = torch.cat(q_states, dim=2)
        k = torch.cat(k_states, dim=2)
        v = torch.cat(v_states, dim=2)
        q, k = self.rotary_emb(position_ids, q, k)

        bsz = q.shape[0]
        attn_outputs = self.attn(q, k, v, attention_mask, use_cache=use_cache)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.view(bsz, -1, self.num_heads * self.head_dim)

        outputs = []
        start_pos = 0
        for attn, hs in zip(attns, hidden_states):
            if hs is None or hs.shape[1] == 0:
                outputs.append(None)
                continue
            end_pos = start_pos + hs.shape[1]
            out_emb = attn.o_proj(attn_outputs[:, start_pos:end_pos])
            outputs.append(out_emb)
            start_pos = end_pos
        return outputs


class PI0MLP(nn.Module):
    """Joint feed-forward network for VLM and action expert."""

    def __init__(self, config: PI0Config, vlm_mlp: nn.Module, action_expert_mlp: nn.Module):
        super().__init__()
        self.config = config
        self.vlm_mlp = vlm_mlp
        self.action_expert_mlp = action_expert_mlp

    def forward(self, hidden_states):
        mlps = [self.vlm_mlp, self.action_expert_mlp]
        outputs = []
        for mlp, hs in zip(mlps, hidden_states):
            if hs is None or hs.shape[1] == 0:
                outputs.append(hs)
                continue
            if hasattr(mlp, "gate_up_proj"):
                gate, up = mlp.gate_up_proj(hs)
            else:
                gate = mlp.gate_proj(hs)
                up = mlp.up_proj(hs)
            x = mlp.act_fn(gate) * up
            x = mlp.down_proj(x)
            outputs.append(x)
        return outputs


class PI0ModelLayer(nn.Module):
    """Single shared transformer layer pairing one VLM layer and one action-expert layer."""

    def __init__(
        self,
        config: PI0Config,
        vlm_layer: nn.Module,
        action_expert_layer: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.input_layernorm = [vlm_layer.input_layernorm, action_expert_layer.input_layernorm]
        self.post_attention_layernorm = [
            vlm_layer.post_attention_layernorm,
            action_expert_layer.post_attention_layernorm,
        ]
        self.self_attn = PI0Attention(config, vlm_layer.self_attn, action_expert_layer.self_attn)
        self.mlp = PI0MLP(config, vlm_layer.mlp, action_expert_layer.mlp)

    def forward(self, hidden_states, attention_mask, position_ids, conds, use_cache: bool = False):
        # Pre-attention norm + gated residual
        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                gates.append(None)
                continue
            hidden_states[i], gate = layernorm_forward(self.input_layernorm[i], hs, conds[i])
            gates.append(gate)

        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, conds, use_cache=use_cache)

        for i in range(len(hidden_states)):
            if hidden_states[i] is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hidden_states[i], gates[i])

        # Pre-MLP norm + gated residual
        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                gates.append(None)
                continue
            hidden_states[i], gate = layernorm_forward(self.post_attention_layernorm[i], hs, conds[i])
            gates.append(gate)

        hidden_states = self.mlp(hidden_states)

        for i in range(len(hidden_states)):
            if hidden_states[i] is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hidden_states[i], gates[i])

        return hidden_states


class PI0Model(nn.Module):
    """Core PI0 model implementing flow matching for action generation.

    Combines a PaliGemma VLM backbone with a Gemma action expert through
    shared transformer layers.  Training uses flow matching; inference uses
    Euler integration (t: 1 → 0).
    """

    def __init__(
        self,
        config: PI0Config,
        _skip_dtype_conversion: bool = False,
        _skip_random_init: bool = False,
    ):
        super().__init__()
        self.config = config

        # VLM backbone + action expert.
        # When loading from a checkpoint, every parameter is overwritten by
        # load_state_dict, so HuggingFace's default random weight init (a pass
        # over ~3.5B params, ~75s on CPU) is pure waste. Skip it with
        # no_init_weights — tensors are still allocated, just left uninitialised
        # until the checkpoint fills them.
        with no_init_weights() if _skip_random_init else contextlib.nullcontext():
            self.vlm = PaliGemmaForConditionalGeneration(config.vlm_config)
            self.action_expert = GemmaForCausalLM(config.action_expert_config)

        # Embedders
        self.prefix_embedder = PI0PrefixEmbedder(config, self.vlm)
        self.suffix_embedder = PI0SuffixEmbedder(config)

        # Shared transformer layers (one per backbone layer)
        num_layers = config.vlm_config.text_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                PI0ModelLayer(
                    config,
                    self.vlm.model.language_model.layers[i],
                    self.action_expert.model.layers[i],
                )
                for i in range(num_layers)
            ]
        )

        # Output projection: action expert hidden → action space
        self.action_out_proj = nn.Linear(config.action_expert_config.hidden_size, config.max_action_dim)

        # Real-Time Chunking (RTC) guidance processor. Left as None (RTC off) unless
        # a deployment attaches a lerobot ``RTCProcessor`` here (see robot_inference).
        # RTC is a training-free, inference-time flow-matching inpainting scheme that
        # keeps successive action chunks continuous under inference latency.
        self.rtc_processor = None

        # Skip dtype conversion when loading from checkpoint — caller does it after load_state_dict
        # so we avoid converting random init weights that will be overwritten anyway.
        if not _skip_dtype_conversion:
            self.to_bfloat16_for_selected_params(getattr(config, "dtype", "float32"))

        # Keep the uncompiled bound methods so the compile target can be switched
        # later (e.g. when RTC is attached at deploy time) without stacking wrappers.
        self._uncompiled_sample_actions = self.sample_actions
        self._uncompiled_denoise_step = self.denoise_step
        self._apply_compile()

    def _apply_compile(self) -> None:
        """(Re)apply ``torch.compile`` to the right target given the current RTC state.

        No RTC → compile the whole ``sample_actions`` (one fused inference graph,
        fastest). RTC on → compiling ``sample_actions`` would graph-break at the
        per-step ``torch.autograd.grad`` inside the ΠGDM guidance and churn
        recompiles, so instead compile only the per-step ``denoise_step``: the heavy
        transformer forward stays fused while the guided loop and its vector-Jacobian
        product run in eager, with autograd flowing through the compiled step via
        AOTAutograd. Always resets to the uncompiled bound methods first so the target
        can be switched (e.g. when RTC is attached after construction) without stacking
        wrappers. Call again whenever ``rtc_processor``/``compile_model`` changes.
        """
        # Reset to a clean, uncompiled baseline so switching targets is idempotent.
        self.sample_actions = self._uncompiled_sample_actions
        self.denoise_step = self._uncompiled_denoise_step
        if not self.config.compile_model:
            return

        torch.set_float32_matmul_precision("high")
        if self._rtc_enabled():
            # RTC differentiates through this step (AOTAutograd compiles a *backward*
            # graph too). "max-autotune" would then autotune every forward AND backward
            # matmul — minutes of compile that stalls the first guided control step.
            # Force a light mode: steady-state kernels stay fast, compile is seconds.
            rtc_mode = "default" if self.config.compile_mode == "max-autotune" else self.config.compile_mode
            self.denoise_step = torch.compile(
                self.denoise_step, mode=rtc_mode, dynamic=False,
            )
            logger.info("PI0 torch.compile target: per-step denoise_step (RTC-compatible, mode=%s).", rtc_mode)
        elif self._ttrtc_enabled():
            self.denoise_step = torch.compile(
                self.denoise_step, mode=self.config.compile_mode, dynamic=False,
            )
            logger.info("PI0 torch.compile target: per-step denoise_step (TTRTC-compatible).")
        else:
            self.sample_actions = torch.compile(
                self.sample_actions, mode=self.config.compile_mode, dynamic=False,
            )
            logger.info("PI0 torch.compile target: sample_actions (full sampler).")

    # ------------------------------------------------------------------
    # Weight fusion helpers (for faster inference)
    # ------------------------------------------------------------------

    def init_qkv_fusion_from_existing(self) -> None:
        """Fuse Q/K/V projections into a single QKVLinear for efficiency."""
        backbones = [self.vlm.model.language_model, self.action_expert.model]
        for backbone in backbones:
            for idx in range(backbone.config.num_hidden_layers):
                attn = backbone.layers[idx].self_attn
                q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
                hidden_size = q_proj.in_features
                head_dim = attn.head_dim
                num_heads = self.vlm.model.language_model.config.num_attention_heads
                num_kv_heads = self.vlm.model.language_model.config.num_key_value_heads

                qkv = QKVLinear(hidden_size, head_dim, num_heads, num_kv_heads, bias=q_proj.bias is not None)
                attn.qkv_proj = qkv
                qkv.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)

                with torch.no_grad():
                    q_span = num_heads * head_dim
                    kv_span = num_kv_heads * head_dim
                    qkv.weight[:q_span].copy_(q_proj.weight)
                    qkv.weight[q_span : q_span + kv_span].copy_(k_proj.weight)
                    qkv.weight[q_span + kv_span :].copy_(v_proj.weight)
                    if qkv.bias is not None:
                        qkv.bias[:q_span].copy_(q_proj.bias)
                        qkv.bias[q_span : q_span + kv_span].copy_(k_proj.bias)
                        qkv.bias[q_span + kv_span :].copy_(v_proj.bias)

                delattr(attn, "q_proj")
                delattr(attn, "k_proj")
                delattr(attn, "v_proj")

    def init_mlp_fusion_from_existing(self) -> None:
        """Fuse gate/up projections into a single MergedColumnLinear."""
        backbones = [self.vlm.model.language_model, self.action_expert.model]
        for backbone in backbones:
            for idx in range(backbone.config.num_hidden_layers):
                mlp = backbone.layers[idx].mlp
                inter = mlp.intermediate_size
                gate_up = MergedColumnLinear(mlp.hidden_size, [inter, inter], bias=False)
                mlp.gate_up_proj = gate_up
                gate_up.to(device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype)

                with torch.no_grad():
                    gate_up.weight[:inter].copy_(mlp.gate_proj.weight)
                    gate_up.weight[inter:].copy_(mlp.up_proj.weight)

                delattr(mlp, "gate_proj")
                delattr(mlp, "up_proj")

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        """Convert model to bfloat16 while keeping vision tower and norms in float32.

        The entire SigLIP vision tower and multi_modal_projector stay float32 so
        that the dtype is consistent end-to-end through the image encoder (mixing
        float32 patch embeddings with bfloat16 LayerNorm weights causes a crash).
        """
        if precision == "float32":
            for m in [self.vlm, self.action_expert]:
                m.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        float32_selectors = (
            "vision_tower",           # entire SigLIP encoder stays float32
            "multi_modal_projector",  # vision→language projection stays float32
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, param in self.named_parameters():
            target = torch.float32 if any(s in name for s in float32_selectors) else torch.bfloat16
            if param.data.dtype != target:
                param.data = param.data.to(dtype=target)

    # ------------------------------------------------------------------
    # Shared layer stack (training) with optional gradient checkpointing
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_checkpoint(layer, prefix_h, suffix_h, attention_mask, position_ids):
        """Functional wrapper around a layer for torch.utils.checkpoint.

        Takes the two hidden-state tensors as separate args (so checkpoint never
        sees / mutates a shared list) and rebuilds the list inside.  PI0 has no
        adaRMS conditioning, so ``conds`` is always ``[None, None]``.
        """
        out = layer([prefix_h, suffix_h], attention_mask, position_ids, [None, None], use_cache=False)
        return out[0], out[1]

    def _run_layers(self, prefix_embs, suffix_embs, attention_mask, position_ids):
        """Run all shared transformer layers, optionally with gradient checkpointing.

        When ``config.gradient_checkpointing`` is set and the module is in training
        mode, each layer is wrapped in non-reentrant ``checkpoint`` so its
        intermediate activations are recomputed during backward instead of being
        kept in memory — trading ~20-30% extra compute for a large activation-memory
        saving.  At eval/inference it is a plain layer loop with no overhead.
        """
        hidden_states = [prefix_embs, suffix_embs]
        use_ckpt = getattr(self.config, "gradient_checkpointing", False) and self.training
        for layer in self.layers:
            if use_ckpt:
                prefix_h, suffix_h = checkpoint(
                    self._layer_checkpoint,
                    layer, hidden_states[0], hidden_states[1],
                    attention_mask, position_ids,
                    use_reentrant=False,
                    determinism_check="none",
                )
                hidden_states = [prefix_h, suffix_h]
            else:
                hidden_states = layer(
                    hidden_states, attention_mask, position_ids, [None, None], use_cache=False
                )
        return hidden_states

    # ------------------------------------------------------------------
    # Noise / time sampling
    # ------------------------------------------------------------------

    def sample_noise(self, shape, device):
        """Sample standard Gaussian noise for flow matching."""
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        """Sample flow-matching timestep from Beta(1.5, 1.0) scaled to (0, 1)."""
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        t = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return t * 0.999 + 0.001

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, images, img_masks, tokens, masks, state, actions, noise=None, time=None):
        """Training forward pass: compute flow-matching velocity loss.

        Args:
            images: List of image tensors [B, C, H, W].
            img_masks: List of boolean masks [B].
            tokens: Language token IDs [B, L_text].
            masks: Language attention masks [B, L_text].
            state: Robot state [B, state_dim].
            actions: Ground-truth actions [B, T, action_dim].
            noise: Optional Gaussian noise (sampled if None).
            time: Optional flow-matching timestep (sampled if None).

        Returns:
            losses: Per-element MSE loss [B, T, action_dim].
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(state, x_t, time)

        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        attention_mask, position_ids = build_attention_mask_and_position_ids(
            torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1),
            torch.cat([prefix_att_masks, suffix_att_masks], dim=1),
            prefix_embs.dtype,
        )

        conds = [None, suffix_adarms_cond]
        hidden_states = self._run_layers(prefix_embs, suffix_embs, attention_mask, position_ids)

        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        final_hidden_states = []
        for i, hs in enumerate(hidden_states):
            if hs is None:
                final_hidden_states.append(None)
                continue
            hs, _ = layernorm_forward(norms[i], hs, conds[i])
            final_hidden_states.append(hs)
        hidden_states = final_hidden_states

        suffix_out = hidden_states[1][:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)

        # Loss is computed under accelerator.autocast(): AMP runs mse_loss in
        # float32 with autograd-aware casts, so no manual dtype reconciliation is
        # needed here regardless of the (bfloat16) prediction dtype.
        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_ttrtc(self, images, img_masks, tokens, masks, state, actions, noise=None, time=None):
        """Training forward pass for TTRTC (Training-Time Real-Time Chunking).

        Gives every action token its own flow-matching timestep: a random prefix of
        length ``d ~ U{0..min(ttrtc_max_delay, chunk_size)}`` is treated as a clean,
        known future (hard-inpainted with the ground-truth actions at ``t=0``) while
        the remaining postfix is the usual noisy target at the sampled ``t``. Only
        the postfix contributes to the loss; the prefix is pure conditioning. This
        mirrors the inference-time hard clamp in ``sample_actions`` so the previous
        chunk pins the new chunk exactly, with no extra inference cost.

        The attention mask is byte-for-byte identical to ``forward`` — the prefix is
        injected purely through token content (clean action value + ``t=0``).

        Returns:
            losses: Per-element MSE loss [B, T, action_dim] (unmasked).
            prefix_mask: Boolean [B, T]; True where the token is a clean prefix
                (excluded from the loss by the caller).
        """
        bsize = actions.shape[0]
        horizon = actions.shape[1]
        device = actions.device

        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(bsize, device)  # [B]

        # Per-sample prefix length d ~ U{0..max_delay} (inclusive), clamped to horizon.
        max_delay = min(int(self.config.ttrtc_max_delay), horizon)
        delay = torch.randint(0, max_delay + 1, (bsize,), device=device)  # [B]
        prefix_mask = torch.arange(horizon, device=device)[None, :] < delay[:, None]  # [B, T]

        # Postfix is the standard flow interpolation; prefix is hard-inpainted with GT.
        time_expanded = time[:, None, None]
        x_t_postfix = time_expanded * noise + (1 - time_expanded) * actions
        x_t = torch.where(prefix_mask[:, :, None], actions, x_t_postfix)
        # Per-token timestep: prefix tokens are clean (t=0), postfix uses sampled t.
        token_time = torch.where(prefix_mask, torch.zeros_like(time[:, None]), time[:, None])  # [B, T]
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, token_time
        )

        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        attention_mask, position_ids = build_attention_mask_and_position_ids(
            torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1),
            torch.cat([prefix_att_masks, suffix_att_masks], dim=1),
            prefix_embs.dtype,
        )

        conds = [None, suffix_adarms_cond]
        hidden_states = self._run_layers(prefix_embs, suffix_embs, attention_mask, position_ids)

        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        final_hidden_states = []
        for i, hs in enumerate(hidden_states):
            if hs is None:
                final_hidden_states.append(None)
                continue
            hs, _ = layernorm_forward(norms[i], hs, conds[i])
            final_hidden_states.append(hs)
        hidden_states = final_hidden_states

        suffix_out = hidden_states[1][:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none"), prefix_mask

    def forward_shared_observation(
        self,
        images,
        img_masks,
        tokens,
        masks,
        states,
        actions,
        offset_mask,
        noise=None,
        time=None,
    ):
        """Training forward pass with shared observation across multiple time offsets.

        Computes prefix embeddings once and shares them across all offset branches,
        significantly reducing compute when training with multiple temporal offsets.

        Returns:
            losses: Per-element MSE loss [B, num_offsets, T, action_dim].
        """
        batch_size, num_offsets = states.shape[:2]
        device = states.device

        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(batch_size * num_offsets, device).view(batch_size, num_offsets)

        time_expanded = time[:, :, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(images, img_masks, tokens, masks)
        prefix_length = prefix_embs.shape[1]

        states_flat = states.view(batch_size * num_offsets, -1)
        x_t_flat = x_t.view(batch_size * num_offsets, x_t.shape[2], -1)
        time_flat = time.view(batch_size * num_offsets)

        suffix_embs_flat, suffix_pad_masks_flat, suffix_att_masks_flat, _ = self.suffix_embedder(
            states_flat, x_t_flat, time_flat
        )
        suffix_length = suffix_embs_flat.shape[1]

        suffix_pad_masks = suffix_pad_masks_flat[:batch_size]
        suffix_att_masks = suffix_att_masks_flat[:batch_size]

        suffix_embs = suffix_embs_flat.view(batch_size, num_offsets, suffix_length, -1)
        suffix_embs_concat = suffix_embs.view(batch_size, num_offsets * suffix_length, -1)

        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs_concat = suffix_embs_concat.to(dtype=backbone_dtype)

        attention_mask, position_ids = build_shared_obs_attention_mask_and_position_ids(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            num_offsets=num_offsets,
            offset_mask=offset_mask,
            dtype=prefix_embs.dtype,
        )

        conds = [None, None]
        hidden_states = self._run_layers(prefix_embs, suffix_embs_concat, attention_mask, position_ids)

        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        final_hidden_states = []
        for i, hs in enumerate(hidden_states):
            if hs is None:
                final_hidden_states.append(None)
                continue
            hs, _ = layernorm_forward(norms[i], hs, conds[i])
            final_hidden_states.append(hs)
        hidden_states = final_hidden_states

        suffix_out = hidden_states[1].view(batch_size, num_offsets, suffix_length, -1)
        action_out = suffix_out[:, :, -self.config.chunk_size :, :]
        action_out = action_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(action_out)

        # See note in forward(): mse_loss runs in float32 under autocast.
        return F.mse_loss(u_t, v_t, reduction="none")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _rtc_enabled(self) -> bool:
        """True when an enabled RTC guidance processor is attached."""
        proc = getattr(self, "rtc_processor", None)
        return proc is not None and getattr(proc.rtc_config, "enabled", False)

    def _ttrtc_enabled(self) -> bool:
        """True when this checkpoint was trained with TTRTC (per-token flow prefix).

        Unlike RTC, TTRTC needs no attached processor — the hard-clamp path is
        entirely inside ``sample_actions`` and gated purely on the config flag.
        """
        return bool(getattr(self.config, "ttrtc", False))

    def _ttrtc_prepare_prefix(
        self, prev_chunk_left_over, inference_delay, execution_horizon, target_shape, device
    ):
        """Prepare the TTRTC inference prefix: pad the previous chunk to the model's
        action space and compute the hard-clamp length ``d``.

        Mirrors kai0 ``pi0_ttrtc`` ``compute_runtime_prefix_steps``:
        ``d = clamp(min(inference_delay, execution_horizon, chunk_size, prev_len), 0)``.
        ``inference_delay=None`` (first chunk / cold start) ⇒ ``d=0`` ⇒ plain pi0
        sampling.

        Returns:
            prev: [B, chunk_size, max_action_dim] float32 (zero-padded).
            d: int hard-clamp prefix length.
        """
        bsz, horizon, adim = target_shape
        prev = prev_chunk_left_over
        if prev.dim() < 3:
            prev = prev.unsqueeze(0)
        prev = prev.to(device=device, dtype=torch.float32)
        if prev.shape[0] == 1 and bsz > 1:
            prev = prev.expand(bsz, -1, -1)

        valid_len = min(prev.shape[1], horizon)
        if prev.shape[0] != bsz or prev.shape[1] != horizon or prev.shape[2] != adim:
            padded = torch.zeros(bsz, horizon, adim, dtype=torch.float32, device=device)
            padded[:, :valid_len, : min(prev.shape[2], adim)] = prev[:, :valid_len, :adim]
            prev = padded

        if inference_delay is None:
            d = 0
        else:
            candidates = [int(inference_delay), horizon, valid_len]
            if execution_horizon is not None:
                candidates.append(int(execution_horizon))
            d = max(0, min(candidates))
        return prev, d

    # NOTE: intentionally *not* decorated with @torch.no_grad(). RTC guidance needs
    # a grad graph through this step (it re-enables grad via torch.enable_grad() and
    # takes a vector-Jacobian product). When RTC is off, sample_actions runs under
    # @torch.no_grad(), so this step is still grad-free with no extra cost.
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Single Euler denoising step using cached prefix KV states."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, timestep
        )

        # o_proj exists whether or not QKV fusion was applied (qkv_proj only
        # exists when fuse_qkv=True), so read the backbone dtype from it.
        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        full_attention_mask, full_position_ids = build_attention_mask_and_position_ids(
            pad_masks, att_masks, suffix_embs.dtype
        )

        bsz, L_suf = suffix_embs.shape[:2]
        attention_mask = full_attention_mask[:, :, -L_suf:, :]
        position_ids = full_position_ids[:, -L_suf:]

        hidden_states = [None, suffix_embs]
        conds = [None, suffix_adarms_cond]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, conds, use_cache=True)

        suffix_hidden = hidden_states[1]
        suffix_hidden, _ = layernorm_forward(self.action_expert.model.norm, suffix_hidden, suffix_adarms_cond)

        suffix_out = suffix_hidden[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(
        self, images, img_masks, tokens, masks, state, noise=None, num_steps=None, **kwargs
    ) -> torch.Tensor:
        """Sample an action chunk from noise via Euler integration (t: 1 → 0).

        Args:
            images: List of image tensors [B, C, H, W].
            img_masks: List of validity masks [B].
            tokens: Language token IDs [B, L].
            masks: Language attention masks [B, L].
            state: Robot state [B, state_dim].
            noise: Optional starting noise; sampled if None.
            num_steps: Euler integration steps; uses config default if None.
            **kwargs: Optional Real-Time Chunking (RTC) guidance inputs, forwarded
                when an enabled ``rtc_processor`` is attached:
                ``prev_chunk_left_over`` (normalised leftover of the previous chunk,
                ``[B, T_prev, action_dim]``), ``inference_delay`` (frozen-prefix
                length ``d``) and ``execution_horizon`` (``s``). When
                ``prev_chunk_left_over`` is None (first chunk), guidance is skipped.

        Returns:
            Predicted action chunk [B, chunk_size, max_action_dim].
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsz = tokens.shape[0]
        device = tokens.device

        if noise is None:
            noise = self.sample_noise((bsz, self.config.chunk_size, self.config.max_action_dim), device)

        # Prefill: embed prefix and cache KV states
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )

        # The prefix embeddings inherit the SigLIP vision tower's float32 dtype,
        # but the backbone attention weights are bfloat16; cast to match so the
        # prefill F.linear calls see consistent dtypes (mirrors forward() and
        # denoise_step()). Read the dtype from o_proj, which exists whether or not
        # QKV fusion was applied (qkv_proj only exists when fuse_qkv=True).
        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)

        for layer in self.layers:
            layer.self_attn.attn.reset_cache()

        prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
            prefix_pad_masks, prefix_att_masks, prefix_embs.dtype
        )

        # Propagate the prefix through the layers so each layer caches K/V from
        # its *own* input (= the previous layer's output), not the raw embeddings.
        # Reassigning hidden_states each iteration is essential: without it every
        # layer would cache K/V computed from layer-0's input, corrupting the
        # prefix KV cache for all deeper layers and destroying conditioning at
        # inference (training's forward() propagates, so they must match).
        hidden_states_prefill = [prefix_embs, None]
        for layer in self.layers:
            hidden_states_prefill = layer(
                hidden_states_prefill,
                prefix_attention_mask,
                prefix_position_ids,
                [None, None],
                use_cache=True,
            )


        dt_scalar = -1.0 / num_steps
        dt = torch.tensor(dt_scalar, dtype=torch.float32, device=device)
        x_t = noise

        prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
        rtc_active = self._rtc_enabled() and prev_chunk_left_over is not None
        ttrtc_active = self._ttrtc_enabled() and prev_chunk_left_over is not None
        if ttrtc_active:
            ttrtc_prev, ttrtc_d = self._ttrtc_prepare_prefix(
                prev_chunk_left_over,
                inference_delay=kwargs.get("inference_delay"),
                execution_horizon=kwargs.get("execution_horizon"),
                target_shape=x_t.shape,
                device=device,
            )
            prefix_mask = (
                torch.arange(self.config.chunk_size, device=device) < ttrtc_d
            )[None, :]  # [1, T]
            prefix_mask_x = prefix_mask[:, :, None]  # [1, T, 1]
            x_t = torch.where(prefix_mask_x, ttrtc_prev, x_t)  # (1) input inpaint

        for step in range(num_steps):
            time_scalar = 1.0 + step * dt_scalar

            if ttrtc_active:
                # Per-token time: clean prefix stays at t=0, postfix follows the schedule.
                base_time = torch.full(
                    (bsz, self.config.chunk_size), time_scalar, dtype=torch.float32, device=device
                )
                token_time = torch.where(prefix_mask, torch.zeros_like(base_time), base_time)
                v_t = self.denoise_step(prefix_pad_masks, prefix_att_masks, state, x_t, token_time)
                x_t = torch.where(prefix_mask_x, ttrtc_prev, x_t + dt * v_t)  # (2/3) re-clamp
                continue

            time_tensor = torch.full((bsz,), time_scalar, dtype=torch.float32, device=device)

            def denoise_step_partial_call(input_x_t, _timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks, prefix_att_masks, state, input_x_t, _timestep
                )

            if rtc_active:
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=kwargs.get("inference_delay"),
                    time=time_scalar,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

        if ttrtc_active:
            x_t = torch.where(prefix_mask_x, ttrtc_prev, x_t)  # (4) final clamp

        return x_t


class PI0Policy(PreTrainedPolicy):
    """PI0 Policy: System 1 (Generative Executor) of Robo-MLT.

    Wraps PI0Model for lerobot-compatible training and inference.
    Expects batches to be pre-processed by make_pi0_pre_post_processors
    (state/action normalised, language tokenised).
    """

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        _skip_dtype_conversion: bool = False,
        _skip_random_init: bool = False,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = PI0Model(
            config,
            _skip_dtype_conversion=_skip_dtype_conversion,
            _skip_random_init=_skip_random_init,
        )
        self.reset()

    # ------------------------------------------------------------------
    # Loading from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        """Load a PI0Policy from a checkpoint directory.

        Supports both native checkpoints (model.safetensors) and OpenPI-format
        checkpoints via prefix-based key remapping.
        """
        if config is None:
            # Load the config locally (plain JSON) instead of going through
            # lerobot/draccus, which fails to decode the nested HF sub-configs.
            config = PI0Config.from_pretrained_local(pretrained_name_or_path)

        # Skip dtype conversion AND random weight init — every parameter is
        # overwritten by the checkpoint below, so both passes over ~3.5B params
        # are pure waste. Skipping random init alone saves ~75s of CPU build time.
        instance = cls(config, _skip_dtype_conversion=True, _skip_random_init=True, **kwargs)

        from safetensors.torch import load_file
        from transformers.utils import cached_file

        if os.path.isdir(pretrained_name_or_path):
            model_file = os.path.join(pretrained_name_or_path, "model.safetensors")
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"No 'model.safetensors' in: {model_file}")
            original_state_dict = load_file(model_file)
        else:
            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            if resolved_file is None:
                raise FileNotFoundError(f"Could not resolve 'model.safetensors' for {pretrained_name_or_path}")
            original_state_dict = load_file(resolved_file)

        # Key remapping from OpenPI format to our internal naming
        prefix_rules: list[tuple[str, str]] = [
            ("action_in_proj.", "model.suffix_embedder.action_in_proj."),
            ("action_out_proj.", "model.action_out_proj."),
            ("action_time_mlp_in.", "model.suffix_embedder.action_time_mlp_in."),
            ("action_time_mlp_out.", "model.suffix_embedder.action_time_mlp_out."),
            ("state_proj.", "model.suffix_embedder.state_proj."),
            ("paligemma_with_expert.gemma_expert.", "model.action_expert."),
            ("paligemma_with_expert.paligemma.", "model.vlm."),
        ]

        def map_key(key: str) -> str:
            for src, dst in prefix_rules:
                if key.startswith(src):
                    return dst + key[len(src) :]
            return key

        if getattr(config, "fuse_qkv", True):
            instance.model.init_qkv_fusion_from_existing()
        if getattr(config, "fuse_gate_up", True):
            instance.model.init_mlp_fusion_from_existing()

        target_sd = instance.state_dict()
        mapped_sd: dict[str, Tensor] = {}
        for old_key, value in original_state_dict.items():
            new_key = map_key(old_key)
            if new_key in target_sd and target_sd[new_key].shape != value.shape:
                continue
            mapped_sd[new_key] = value

        incompatible = instance.load_state_dict(mapped_sd, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint loading failed. Unexpected keys: {incompatible.unexpected_keys}"
            )

        # Apply dtype conversion once, after the checkpoint is loaded.
        instance.model.to_bfloat16_for_selected_params(getattr(config, "dtype", "float32"))

        instance.to(config.device)
        instance.eval()
        return instance

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset action queue. Call at the start of each new episode."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Predict a full action chunk from a preprocessed batch.

        The batch must have been processed by the preprocessor pipeline
        (language tokenised, state/action normalised).

        Args:
            batch: Preprocessed observation batch.
            noise: Optional starting noise for the flow-matching sampler.
            **kwargs: Optional Real-Time Chunking guidance inputs
                (``prev_chunk_left_over``, ``inference_delay``, ``execution_horizon``),
                forwarded to ``sample_actions``.

        Returns:
            actions: [B, n_action_steps, action_dim] in normalised space.
                     The postprocessor should be applied to unnormalise.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions[:, : self.config.n_action_steps, :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return a single action step using action chunking."""
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._action_queue.popleft()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass.

        Args:
            batch: Pre-processed batch with normalised state/action and
                   tokenised language (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK).
            noise: Optional noise override.
            time: Optional timestep override.

        Returns:
            (loss, loss_dict)
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)

        losses = losses[:, :, : self.config.max_action_dim]
        loss = losses.mean()
        return loss, {"loss": loss.item()}

    def forward_shared_observation(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward with shared observation across temporal offsets."""
        offset_mask = batch["offset_mask"]
        states_normalized = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        actions_normalized = pad_vector(batch[ACTION], self.config.max_action_dim)
        images, img_masks = self.prepare_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward_shared_observation(
            images, img_masks, lang_tokens, lang_masks,
            states_normalized, actions_normalized, offset_mask,
            noise, time,
        )

        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        losses = losses * offset_mask[:, :, None, None]
        losses = losses[:, :, :, : self.config.max_action_dim]

        num_valid = offset_mask.sum()
        n_per_offset = losses.shape[2] * losses.shape[3]
        loss = losses.sum() / (num_valid * n_per_offset).clamp(min=1)

        return loss, {
            "loss": loss.item(),
            "num_offsets": offset_mask.shape[1],
            "avg_valid_offsets": offset_mask.float().sum(dim=1).mean().item(),
        }

    def forward_ttrtc(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward for TTRTC: loss is averaged over the noisy postfix only.

        The clean action prefix (hard-inpainted at ``t=0``) and any padded action
        steps are excluded from the mean.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        losses, prefix_mask = self.model.forward_ttrtc(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        losses = losses[:, :, : self.config.max_action_dim]

        # Supervise only noisy postfix tokens that are also not padding.
        postfix_mask = ~prefix_mask  # [B, T]
        if actions_is_pad is not None:
            postfix_mask = postfix_mask & (~actions_is_pad)
        postfix_mask = postfix_mask[:, :, None]  # [B, T, 1]

        n_dims = losses.shape[-1]
        denom = (postfix_mask.sum() * n_dims).clamp(min=1)
        loss = (losses * postfix_mask).sum() / denom
        return loss, {"loss": loss.item()}

    # ------------------------------------------------------------------
    # Batch preparation helpers
    # ------------------------------------------------------------------

    def prepare_images(self, batch):
        """Extract and preprocess images for SigLIP.

        Converts images to float32 [0,1], pads to image_resolution with center
        crop, and normalises to [-1, 1] as required by SigLIP.
        """
        images: list[Tensor] = []
        img_masks: list[Tensor] = []
        device = next(self.parameters()).device

        present_keys = [k for k in self.config.image_features if k in batch]
        missing_keys = [k for k in self.config.image_features if k not in batch]

        if not present_keys:
            raise ValueError(
                f"All image features missing from batch. "
                f"Expected one of: {list(self.config.image_features.keys())}"
            )

        img = None  # keep reference for empty-camera placeholder
        mask = None

        for key in present_keys:
            img = batch[key].to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32) / 255.0 if img.dtype == torch.uint8 else img.to(torch.float32)
            img = resize_with_pad(img, *self.config.image_resolution, pad_value=0)
            img = img * 2.0 - 1.0
            mask = torch.ones(img.shape[0], dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for i in range(len(missing_keys)):
            if i >= self.config.empty_cameras:
                break
            images.append(torch.ones_like(img) * -1)
            img_masks.append(torch.zeros_like(mask))

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state tensor to max_state_dim."""
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch):
        """Pad action tensor to max_action_dim."""
        return pad_vector(batch[ACTION], self.config.max_action_dim)


__all__ = ["PI0Model", "PI0Policy", "PI0PrefixEmbedder", "PI0SuffixEmbedder", "PI0ModelLayer"]
