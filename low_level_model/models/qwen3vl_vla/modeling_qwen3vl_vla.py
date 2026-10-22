"""Qwen3VL-VLA Policy (System 1 — Generative Executor).

A faithful port of starVLA's ``Qwen_PI`` onto the lerobot ``PreTrainedPolicy``
contract.  The policy couples a Qwen3-VL / Qwen2.5-VL image-text backbone with a
**layer-wise cross-attention flow-matching** action head: the VLM is run with
``output_hidden_states=True`` and the DiT action expert has one transformer block
per VLM hidden state, with block *i* cross-attending to VLM hidden state *i*.

Sources ported:
  - ``starVLA/model/framework/QwenPI.py``                      (Qwen_PI)
  - ``starVLA/model/modules/vlm/QWen3.py``                     (_QWen3_VL_Interface)
  - ``starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py``
  - ``starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py``
  - ``.../flow_matching_head/action_encoder.py``

Differences from the source: the VLM is loaded from a plain ``vlm_model_id``
(auto-selecting the Qwen3-VL or Qwen2.5-VL transformers class), state/action are
zero-padded to ``max_state_dim``/``max_action_dim`` (so one checkpoint serves
varying embodiments, matching PI0), and the flow-matching loss honours lerobot's
``action_is_pad`` mask.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.distributions import Beta

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from low_level_model.models.qwen3vl_vla.configuration_qwen3vl_vla import Qwen3VLVLAConfig

logger = logging.getLogger(__name__)


# ─────────────────────────── action-encoder helpers ──────────────────────────
def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape (B, T, w) given timesteps of shape (B, T)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, output_dim: int = 2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """Encode a noisy action trajectory + per-sample timestep into token embeddings."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor) -> Tensor:
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) to replicate across T.")
        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.W3(x)


# ──────────────────────────────── DiT blocks ─────────────────────────────────
class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: Tensor) -> Tensor:
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, norm_elementwise_affine: bool = False, norm_eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """A single cross-attention DiT block with AdaLayerNorm timestep conditioning."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
    ):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=True,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        temb: Optional[Tensor] = None,
    ) -> Tensor:
        norm_hidden_states = self.norm1(hidden_states, temb)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        ff_output = self.ff(self.norm3(hidden_states))
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


# ───────────────────────── layer-wise flow-matching head ─────────────────────
class LayerwiseFlowMatchingHead(nn.Module):
    """DiT action expert with one cross-attention block per VLM hidden state."""

    def __init__(
        self,
        *,
        num_layers: int,
        hidden_dim: int,
        attention_head_dim: int,
        action_dim: int,
        action_horizon: int,
        state_dim: int,
        num_target_vision_tokens: int,
        num_inference_timesteps: int,
        noise_beta_alpha: float,
        noise_beta_beta: float,
        noise_s: float,
        num_timestep_buckets: int,
        add_pos_embed: bool,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps
        self.noise_s = noise_s
        self.num_timestep_buckets = num_timestep_buckets
        self.add_pos_embed = add_pos_embed

        num_attention_heads = hidden_dim // attention_head_dim
        self.timestep_encoder = TimestepEncoder(hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=hidden_dim,
                    activation_fn="gelu-approximate",
                    attention_bias=True,
                    final_dropout=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.state_encoder = MLP(input_dim=state_dim, output_dim=hidden_dim) if state_dim else None
        self.action_encoder = ActionEncoder(action_dim=action_dim, hidden_size=hidden_dim)
        self.action_decoder = MLP(input_dim=hidden_dim, hidden_dim=1024, output_dim=action_dim)
        self.future_tokens = nn.Embedding(num_target_vision_tokens, hidden_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        if add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)

    # -- helpers ----------------------------------------------------------------
    def _compute_dtype(self) -> torch.dtype:
        return self.action_decoder.layer1.weight.dtype

    def sample_time(self, batch_size: int, device, dtype) -> Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def _build_seq(self, action_features: Tensor, state_features: Optional[Tensor]) -> Tensor:
        B = action_features.shape[0]
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        if state_features is not None:
            return torch.cat((state_features, future_tokens, action_features), dim=1)
        return torch.cat((future_tokens, action_features), dim=1)

    def _run_blocks(self, sa_embs: Tensor, vl_embs_list: list[Tensor], temb: Tensor) -> Tensor:
        model_output = sa_embs
        for layer_idx, layer in enumerate(self.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list[layer_idx],
                temb=temb,
            )
        return model_output

    # -- training ---------------------------------------------------------------
    def forward(self, vl_embs_list: list[Tensor], actions: Tensor, state: Optional[Tensor] = None) -> Tensor:
        """Return the per-element flow-matching squared error, shape (B, T, action_dim)."""
        dtype = self._compute_dtype()
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        actions = actions.to(dtype)
        state = state.to(dtype) if state is not None else None

        noise = torch.randn(actions.shape, device=actions.device, dtype=dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        state_features = self.state_encoder(state) if (state is not None and self.state_encoder) else None

        sa_embs = self._build_seq(action_features, state_features)
        temb = self.timestep_encoder(t_discretized)
        model_output = self._run_blocks(sa_embs, vl_embs_list, temb)

        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]
        return (pred_actions - velocity) ** 2

    # -- inference --------------------------------------------------------------
    @torch.no_grad()
    def predict_action(self, vl_embs_list: list[Tensor], state: Optional[Tensor] = None) -> Tensor:
        dtype = self._compute_dtype()
        vl_embs_list = [h.to(dtype) for h in vl_embs_list]
        state = state.to(dtype) if state is not None else None

        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn((batch_size, self.action_horizon, self.action_dim), dtype=dtype, device=device)
        state_features = self.state_encoder(state) if (state is not None and self.state_encoder) else None

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_int = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            action_features = self.action_encoder(actions, timesteps)
            sa_embs = self._build_seq(action_features, state_features)
            temb = self.timestep_encoder(timesteps)
            model_output = self._run_blocks(sa_embs, vl_embs_list, temb)
            pred_velocity = self.action_decoder(model_output)[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity
        return actions


# ──────────────────────────────── VLM backbone ───────────────────────────────
def _import_qwen3():
    try:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    except Exception:
        return None


def _import_qwen25():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    except Exception:
        return None


def _resolve_vlm_class(model_id: str):
    """Pick the transformers VLM class matching ``model_id`` and what's installed."""
    import transformers

    name = model_id.lower()
    q3, q25 = _import_qwen3(), _import_qwen25()
    if "qwen3" in name:
        if q3 is None:
            raise ImportError(
                f"vlm_model_id={model_id!r} is a Qwen3-VL model, but your transformers "
                f"({transformers.__version__}) lacks Qwen3VLForConditionalGeneration. Upgrade to "
                f"transformers>=4.57, or set vlm_model_id to a Qwen2.5-VL checkpoint."
            )
        return q3
    if "qwen2" in name:
        if q25 is None:
            raise ImportError(
                f"vlm_model_id={model_id!r} is a Qwen2.5-VL model, but Qwen2_5_VLForConditionalGeneration "
                f"is unavailable in transformers {transformers.__version__}."
            )
        return q25
    if q3 is not None:
        return q3
    if q25 is not None:
        return q25
    raise ImportError("Neither Qwen3-VL nor Qwen2.5-VL is available in your transformers install.")


class Qwen3VLVLABackbone(nn.Module):
    """Wraps a Qwen3-VL / Qwen2.5-VL model + processor and exposes per-layer hidden states."""
    def __init__(self, config: Qwen3VLVLAConfig):
        super().__init__()
        from transformers import AutoProcessor

        self.config = config
        torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        vlm_cls = _resolve_vlm_class(config.vlm_model_id)
        self.model = vlm_cls.from_pretrained(
            config.vlm_model_id,
            attn_implementation=config.attn_implementation,
            dtype=torch_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_id)
        self.processor.tokenizer.padding_side = "left"

        # Align Qwen3-VL (which nests dims under text_config) with the flat Qwen2.5 layout.
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = int(getattr(text_config, "hidden_size", self.model.config.hidden_size))
        num_text_layers = int(getattr(text_config, "num_hidden_layers",
                                      getattr(self.model.config, "num_hidden_layers")))
        # One hidden state per layer plus the embedding output → cross-attend one DiT block each.
        self.num_vl_layers = num_text_layers + 1
        self.model.config.hidden_size = self.hidden_size

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @property
    def device(self):
        return self.model.device

    def build_inputs(self, images: list[list[Image.Image]], instructions: list[str]):
        """Build Qwen chat-template inputs from per-sample multi-view PIL images + text."""
        assert len(images) == len(instructions), "images and instructions must align"
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)

    def encode(self, images: list[list[Image.Image]], instructions: list[str]) -> list[Tensor]:
        """Return the list of VLM hidden states (embeddings + per-layer)."""
        inputs = self.build_inputs(images, instructions)
        use_autocast = self.model.device.type == "cuda" and self.config.dtype == "bfloat16"
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else _nullcontext()
        with ctx:
            outputs = self.model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return list(outputs.hidden_states)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ──────────────────────────────────── policy ─────────────────────────────────
def _pad_last_dim(x: Tensor, dim: int) -> Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    return F.pad(x, (0, dim - x.shape[-1]))


class Qwen3VLVLAPolicy(PreTrainedPolicy):
    """Qwen3VL-VLA flow-matching policy (lerobot ``PreTrainedPolicy`` interface)."""

    config_class = Qwen3VLVLAConfig
    name = "qwen3vl_vla"

    def __init__(self, config: Qwen3VLVLAConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.backbone = Qwen3VLVLABackbone(config)
        self.action_head = LayerwiseFlowMatchingHead(
            num_layers=self.backbone.num_vl_layers,
            hidden_dim=self.backbone.hidden_size,
            attention_head_dim=config.attention_head_dim,
            action_dim=config.max_action_dim,
            action_horizon=config.chunk_size,
            state_dim=config.max_state_dim,
            num_target_vision_tokens=config.num_target_vision_tokens,
            num_inference_timesteps=config.num_inference_timesteps,
            noise_beta_alpha=config.noise_beta_alpha,
            noise_beta_beta=config.noise_beta_beta,
            noise_s=config.noise_s,
            num_timestep_buckets=config.num_timestep_buckets,
            add_pos_embed=config.add_pos_embed,
            max_seq_len=config.max_seq_len,
            dropout=config.dit_dropout,
        )

        if config.freeze_vlm:
            self.backbone.model.requires_grad_(False)

        self.reset()

    # -- lerobot interface ------------------------------------------------------
    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        groups = [{"params": list(self.action_head.parameters()), "lr": self.config.optimizer_lr}]
        if not self.config.freeze_vlm:
            vlm_params = [p for p in self.backbone.parameters() if p.requires_grad]
            if vlm_params:
                groups.append({"params": vlm_params, "lr": self.config.vlm_lr})
        return groups

    # -- batch → VLM inputs -----------------------------------------------------
    def _batch_to_pil(self, batch: dict[str, Tensor]) -> list[list[Image.Image]]:
        keys = [k for k in self.config.image_features if k in batch]
        if not keys:
            raise ValueError(
                f"No image features present in batch. Expected one of: "
                f"{list(self.config.image_features.keys())}"
            )
        B = batch[keys[0]].shape[0]
        images_per_sample: list[list[Image.Image]] = [[] for _ in range(B)]
        for k in keys:
            imgs = batch[k]
            for b in range(B):
                arr = (
                    imgs[b].detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
                    .permute(1, 2, 0).cpu().numpy()
                )
                images_per_sample[b].append(Image.fromarray(arr))
        return images_per_sample

    @staticmethod
    def _get_instructions(batch: dict, batch_size: int) -> list[str]:
        task = batch.get("task")
        if task is None:
            return [""] * batch_size
        if isinstance(task, str):
            return [task] * batch_size
        return [str(t) for t in task]

    def _prepare_state(self, batch: dict, ref: Tensor) -> Optional[Tensor]:
        if OBS_STATE not in batch:
            return None
        state = _pad_last_dim(batch[OBS_STATE], self.config.max_state_dim).to(ref.dtype)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (B, 1, state_dim) — one state token
        return state

    def _encode(self, batch: dict) -> list[Tensor]:
        images = self._batch_to_pil(batch)
        instructions = self._get_instructions(batch, len(images))
        vl_embs_list = self.backbone.encode(images, instructions)
        return list(vl_embs_list[-self.backbone.num_vl_layers:])

    # -- training ---------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        vl_embs_list = self._encode(batch)
        ref = vl_embs_list[-1]
        actions = _pad_last_dim(batch[ACTION], self.config.max_action_dim).to(ref.dtype)
        state = self._prepare_state(batch, ref)
        action_is_pad = batch.get("action_is_pad")

        r = max(1, int(self.config.repeated_diffusion_steps))
        vl_rep = [h.repeat(r, 1, 1) for h in vl_embs_list]
        act_rep = actions.repeat(r, 1, 1)
        state_rep = state.repeat(r, 1, 1) if state is not None else None

        sq_err = self.action_head(vl_rep, act_rep, state_rep)  # (r*B, T, action_dim)

        if action_is_pad is not None:
            mask = (~action_is_pad).repeat(r, 1).unsqueeze(-1).to(sq_err.dtype)
            sq_err = sq_err * mask
            loss = sq_err.sum() / mask.expand_as(sq_err).sum().clamp(min=1)
        else:
            loss = sq_err.mean()
        return loss, {"loss": loss.item(), "action_loss": loss.item()}

    # -- inference --------------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        vl_embs_list = self._encode(batch)
        ref = vl_embs_list[-1]
        state = self._prepare_state(batch, ref)
        actions = self.action_head.predict_action(vl_embs_list, state)
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions[:, : self.config.n_action_steps, :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._action_queue.popleft()


__all__ = [
    "Qwen3VLVLAPolicy",
    "Qwen3VLVLABackbone",
    "LayerwiseFlowMatchingHead",
]
