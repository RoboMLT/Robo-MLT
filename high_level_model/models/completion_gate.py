"""Completion gate for System 2 (paper Eq. 3-5).

Two lightweight logistic heads on top of a frozen SigLIP2 temporal trunk:
    - ``completion_head`` (gamma_t): predicts "is this subtask complete?"; the pointer
      controller thresholds its probability to advance the plan pointer.
    - ``back_head`` (beta_t): predicts "has human intervention left the workspace in a state
      from which the active skill can no longer be carried out?"; thresholding triggers
      bounded, protocol-constrained replanning.

Both heads are conditioned on the **current skill text** (frozen SigLIP text embedding), which
keeps them generalizable to unseen skills (no closed-set bottleneck) while reusing one backbone
forward. No second runtime VLM is needed, so VRAM/latency overhead over System 1 is negligible.
"""

import logging
from typing import List, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["CompletionGate"]

SkillText = Union[str, List[str]]


class CausalTemporalEncoder(nn.Module):
    """A small causal Transformer over the per-frame appearance sequence ``[B, T, D]`` (g_psi).

    Each frame attends only to itself and earlier frames (causal mask), so the **last** frame's
    output summarises the whole observed history. Trained jointly with the heads (the SigLIP
    backbone stays frozen).
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8,
                 max_len: int = 16, ffn_mult: int = 4) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(dim)
        # learnable temporal positional embedding (fixed, short horizon → no need for sinusoidal)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * ffn_mult,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.max_len = max_len

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` [B, T, D] → last-frame summary [B, D] under a causal mask."""
        t = seq.shape[1]
        x = self.in_norm(seq) + self.pos[:, :t]
        # bool mask: True = NOT allowed to attend; upper triangle above diagonal → frame i can't see j>i
        attn_mask = torch.triu(seq.new_ones(t, t, dtype=torch.bool), diagonal=1)
        out = self.encoder(x, mask=attn_mask)
        return out[:, -1]  # [B, D] — causal summary up to the current frame


class CrossModalGateFusion(nn.Module):
    """Cross-attention fusion of the **skill text** and the **visual** evidence.

    The skill-text embedding is the *query* and the visual tokens (pooled trunk feature, per-frame
    sequence, causal summary, …) are the *keys / values*: the block pulls the skill-relevant visual
    evidence into the skill query, returning one already-fused vector ``[B, D]``.

    Trained jointly with the heads (the SigLIP backbone stays frozen).
    """

    def __init__(self, dim: int, num_heads: int = 8, ffn_mult: int = 4) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim)
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """``query`` [B, 1, D] (skill text), ``context`` [B, N, D] (visual tokens) → fused [B, D]."""
        kv = self.kv_norm(context)
        attn_out, _ = self.cross_attn(self.q_norm(query), kv, kv)
        h = query + attn_out                       # residual on the skill query → keeps skill identity
        h = h + self.ffn(self.ffn_norm(h))
        return h.squeeze(1)                         # [B, D]


class CompletionGate(nn.Module):
    """Completion + back heads (gamma_t, beta_t) over a frozen SigLIP2 temporal trunk.

    The backbone (a :class:`~high_level_model.models.siglip_encoder.SigLIPFrameEncoder`) is
    frozen by default (``freeze_siglip=True``, matching the paper); set it to ``False`` to
    fine-tune the SigLIP tower.
    """

    def __init__(
        self,
        backbone,
        feature_dim: int = None,
        text_dim: int = None,
        hidden_dim: int = 512,
        freeze_siglip: bool = True,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_max_len: int = 16,
        use_cross_attention: bool = False,
        cross_attn_heads: int = 8,
        cache_text_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        fd = feature_dim or getattr(backbone, "vision_output_dim", 768)
        td = text_dim or fd
        self.feature_dim = fd
        self.text_dim = td

        self.temporal_encoder = CausalTemporalEncoder(
            fd, num_layers=temporal_layers, num_heads=temporal_heads, max_len=temporal_max_len)
        visual_dim = fd + fd  # [pooled_trunk, causal_temporal_summary]

        self.use_cross_attention = bool(use_cross_attention)
        if self.use_cross_attention:
            self.cross_fusion = CrossModalGateFusion(fd, num_heads=cross_attn_heads)
            in_dim = fd + fd  # [fused_skill_visual, aux]
        else:
            in_dim = visual_dim + td

        self.back_head = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.completion_head = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # The frozen SigLIP tower is the whole backbone now (no fusion trunk to separately
        # freeze), so there is a single switch.
        self.freeze_siglip = bool(freeze_siglip)
        if hasattr(self.backbone, "set_siglip_trainable"):
            self.backbone.set_siglip_trainable(not self.freeze_siglip)
        self.cache_text_embeddings = bool(cache_text_embeddings and self.freeze_siglip)
        self._text_cache: dict = {}

    # ------------------------------------------------------------------ text encoder
    def _encode_text_uncached(self, texts: List[str], device) -> torch.Tensor:
        tokens = self.backbone.tokenizer(list(texts)).to(device)
        with torch.no_grad():
            emb = self.backbone.siglip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float()

    def encode_skill_text(self, skill_text: SkillText) -> torch.Tensor:
        """Frozen SigLIP text embedding for the current skill(s) — h_theta(l_t). Returns
        [B, text_dim].

        When ``cache_text_embeddings`` is on, each unique skill string is encoded once and reused,
        so the per-batch tokenise + SigLIP text forward is skipped after warm-up.
        """
        if isinstance(skill_text, str):
            skill_text = [skill_text]
        device = next(self.parameters()).device
        if not self.cache_text_embeddings:
            return self._encode_text_uncached(skill_text, device)
        # dict.fromkeys de-dupes while preserving order
        missing = [s for s in dict.fromkeys(skill_text) if s not in self._text_cache]
        if missing:
            emb = self._encode_text_uncached(missing, device)
            for s, e in zip(missing, emb):
                self._text_cache[s] = e.detach()
        return torch.stack([self._text_cache[s] for s in skill_text], dim=0).to(device)

    # ------------------------------------------------------------------ visual encoder (f_phi + pooling)
    def _visual_outputs_from_siglip(self, feats_tc: torch.Tensor, num_pad_frames: int = 0):
        """Pool frozen SigLIP frame features ``[B, Tp, Cams, D]`` into ``(pooled [B,D],
        frame_seq [B,T,D])``: cameras are averaged per frame and padded frames are dropped from
        the causal sequence (the causal encoder must only see real frames)."""
        frame_seq = feats_tc.float().mean(dim=2)                     # [B, Tp, D] — average cameras
        if num_pad_frames:
            frame_seq = frame_seq[:, num_pad_frames:]                # causal encoder sees real frames only
        pooled = frame_seq.mean(dim=1)                                # [B, D] — window mean
        return pooled, frame_seq

    def _backbone_visual_outputs(self, images: torch.Tensor):
        """Frozen SigLIP frame features -> pooled + frame-sequence outputs. Always runs under
        ``no_grad``: the backbone is frozen (paper: "the SigLIP2 image and text encoders remain
        frozen"), so nothing here needs a graph."""
        with torch.no_grad():
            feats_tc, num_pad = self.backbone.encode_siglip_frames(images)
        return self._visual_outputs_from_siglip(feats_tc, num_pad)

    def _visual_from_backbone_outputs(self, outputs) -> torch.Tensor:
        """Apply the trainable temporal encoder (g_psi) on top of the pooled/frame-sequence
        backbone outputs and concatenate: [pooled_trunk, causal_temporal_summary]."""
        pooled, frame_seq = outputs
        temporal = self.temporal_encoder(frame_seq.float())   # trainable, grad flows here
        return torch.cat([pooled.float(), temporal], dim=-1)

    def _visual_tokens_from_backbone_outputs(self, outputs):
        """Visual token set ``context [B, N, D]`` (keys/values for cross-attention) plus an
        ``aux`` vector ``[B, D]`` (the causal summary) concatenated to the fused output."""
        pooled, frame_seq = outputs
        pooled, frame_seq = pooled.float(), frame_seq.float()
        summary = self.temporal_encoder(frame_seq)                    # [B, D], trainable
        context = torch.cat([pooled.unsqueeze(1), frame_seq, summary.unsqueeze(1)], dim=1)
        return context, summary

    def _heads_input(self, outputs, skill_text_emb: torch.Tensor) -> torch.Tensor:
        """Single assembly point for the head input, shared by :meth:`forward` and
        :meth:`forward_from_cached`. Either cross-attention fusion or the default concat."""
        txt = skill_text_emb.float()
        if self.use_cross_attention:
            context, aux = self._visual_tokens_from_backbone_outputs(outputs)
            if txt.shape[0] == 1 and context.shape[0] > 1:
                txt = txt.expand(context.shape[0], -1)
            fused = self.cross_fusion(txt.unsqueeze(1), context)          # [B, D]
            return torch.cat([fused, aux], dim=-1)
        feat = self._visual_from_backbone_outputs(outputs)
        if txt.shape[0] == 1 and feat.shape[0] > 1:
            txt = txt.expand(feat.shape[0], -1)
        return torch.cat([feat, txt], dim=-1)

    @property
    def cacheable_stage(self):
        """``"frames"`` when the SigLIP tower is frozen (the per-frame SigLIP features are the
        whole frozen stage and can be cached per global frame, letting the history window and
        its random stride be re-gathered every epoch); ``None`` when it trains."""
        return "frames" if self.freeze_siglip else None

    @torch.no_grad()
    def extract_cacheable_features(self, images: torch.Tensor):
        """Frozen-stage outputs to cache, plus the (constant) pad-frame count. Returns
        ``((siglip_frame_features,), num_pad_frames)``. Only valid when :pyattr:`cacheable_stage`
        is not None."""
        feats_tc, num_pad = self.backbone.encode_siglip_frames(images)
        return (feats_tc,), num_pad

    def forward_from_cached(self, cached_outputs, current_skill_text: SkillText,
                            num_pad_frames: int = 0
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the trainable path (temporal encoder + heads) on precomputed frozen SigLIP frame
        features. Returns ``(completion_logit[B], back_logit[B])``."""
        (siglip_feats,) = cached_outputs
        outputs = self._visual_outputs_from_siglip(siglip_feats, num_pad_frames)
        txt = self.encode_skill_text(current_skill_text)
        x = self._heads_input(outputs, txt)
        completion_logit = self.completion_head(x).squeeze(-1)
        back_logit = self.back_head(x).squeeze(-1)
        return completion_logit, back_logit

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        images: torch.Tensor,
        current_skill_text: SkillText,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``images`` [B, M, Cams, C, H, W] -> ``(completion_logit[B], back_logit[B])``."""
        outputs = self._backbone_visual_outputs(images)
        txt = self.encode_skill_text(current_skill_text)              # [B', td]
        x = self._heads_input(outputs, txt)
        completion_logit = self.completion_head(x).squeeze(-1)        # [B]
        back_logit = self.back_head(x).squeeze(-1)                    # [B]
        return completion_logit, back_logit

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, current_skill_text: SkillText
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: returns ``(gamma_t = completion_prob[B], beta_t = back_prob[B])``,
        both in [0, 1]."""
        was_training = self.training
        self.eval()
        completion_logit, back_logit = self.forward(images, current_skill_text)
        if was_training:
            self.train()
        return torch.sigmoid(completion_logit), torch.sigmoid(back_logit)
