"""System 2 execution monitor network (paper Sec. III-B, Fig. 3, Eq. 3-4).

Frozen SigLIP2 encoders embed the M sampled multiview observations (f_phi) and the active skill
instruction (h_theta). The trainable skill-grounding trunk g_psi then
    1. mixes the per-frame features along time with a causal temporal encoder that has learned
       positional encodings, and
    2. grounds that history in the instruction with cross-attention blocks whose query is the text
       embedding and whose keys/values are the temporal features,
returning a single skill-grounded vector v_t. Two unshared two-layer MLP heads read v_t:
    - ``completion_head`` -> gamma_t: has the active skill finished?
    - ``back_head``       -> beta_t (the recovery head): can the active skill still be completed?
The attribute name ``back_head`` matches the dataset label column ``back_event`` and keeps old
checkpoints loadable.

``fusion`` selects the trunk. ``"cross_attention"`` is the architecture described in the paper;
``"concat"`` (window mean + causal summary + text) and ``"legacy_cross_attention"`` (one block
over pooled/frames/summary tokens) only exist to load checkpoints trained before the release.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["CompletionGate", "FUSION_MODES", "gate_kwargs_from_config"]

SkillText = Union[str, List[str]]
FUSION_MODES = ("cross_attention", "concat", "legacy_cross_attention")


class CausalTemporalEncoder(nn.Module):
    """A small causal Transformer over the per-frame feature sequence ``[B, T, D]``.

    Each frame attends only to itself and earlier frames, so the last position summarises the
    whole window. ``forward(..., return_sequence=True)`` returns every position instead (the keys
    and values of the cross-attention blocks).
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8,
                 max_len: int = 16, ffn_mult: int = 4) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)  # learned positional encoding
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * ffn_mult,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.max_len = max_len

    def forward(self, seq: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """``seq`` [B, T, D] -> ``[B, T, D]`` (``return_sequence``) or the last position ``[B, D]``."""
        t = seq.shape[1]
        if t > self.max_len:
            raise ValueError(f"History of {t} frames exceeds temporal_max_len={self.max_len}")
        x = self.in_norm(seq) + self.pos[:, :t]
        # True = not allowed to attend: frame i cannot see frames j > i.
        attn_mask = torch.triu(seq.new_ones(t, t, dtype=torch.bool), diagonal=1)
        out = self.encoder(x, mask=attn_mask)
        return out if return_sequence else out[:, -1]


class CrossModalGateFusion(nn.Module):
    """One cross-attention block: the skill query attends over visual tokens (pre-norm + FFN)."""

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
        """``query`` [B, 1, D], ``context`` [B, N, D] -> updated query [B, 1, D]."""
        kv = self.kv_norm(context)
        attn_out, _ = self.cross_attn(self.q_norm(query), kv, kv)
        h = query + attn_out
        return h + self.ffn(self.ffn_norm(h))


def gate_kwargs_from_config(model_cfg: Optional[dict]) -> dict:
    """Constructor kwargs from the ``model`` block of a checkpoint sidecar / training config.

    Checkpoints written before ``fusion`` existed carry ``use_cross_attention`` (or nothing at
    all), which both map to their original architecture rather than to the current default.
    """
    mc = dict(model_cfg or {})
    if "fusion" in mc:
        fusion = mc["fusion"]
    else:
        fusion = "legacy_cross_attention" if mc.get("use_cross_attention") else "concat"
    return {
        "fusion": fusion,
        "temporal_layers": mc.get("temporal_layers", 2),
        "temporal_heads": mc.get("temporal_heads", 8),
        "cross_attn_layers": mc.get("cross_attn_layers", 2),
        "cross_attn_heads": mc.get("cross_attn_heads", 8),
    }


def _mlp_head(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
    )


class CompletionGate(nn.Module):
    """System 2 monitor: frozen SigLIP2 encoders + trainable trunk g_psi + gamma/beta heads.

    Args:
        backbone: a :class:`~high_level_model.models.siglip_encoder.SigLIPFrameEncoder`.
        fusion: ``"cross_attention"`` (paper), ``"concat"`` or ``"legacy_cross_attention"``.
        temporal_layers / temporal_heads: depth / heads of the causal temporal encoder.
        cross_attn_layers / cross_attn_heads: number / heads of the cross-attention blocks.
        use_cross_attention: deprecated boolean from pre-release checkpoints; when given it maps
            to ``"legacy_cross_attention"`` (True) or ``"concat"`` (False).
    """

    def __init__(
        self,
        backbone,
        feature_dim: int = None,
        text_dim: int = None,
        hidden_dim: int = 512,
        freeze_siglip: bool = True,
        fusion: str = "cross_attention",
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_max_len: int = 16,
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,
        use_cross_attention: Optional[bool] = None,
        cache_text_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if use_cross_attention is not None:
            fusion = "legacy_cross_attention" if use_cross_attention else "concat"
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got '{fusion}'")
        self.fusion = fusion
        self.backbone = backbone
        fd = feature_dim or getattr(backbone, "vision_output_dim", 768)
        td = text_dim or fd
        self.feature_dim = fd
        self.text_dim = td

        self.temporal_encoder = CausalTemporalEncoder(
            fd, num_layers=temporal_layers, num_heads=temporal_heads, max_len=temporal_max_len)

        if fusion == "cross_attention":
            self.text_proj = nn.Identity() if td == fd else nn.Linear(td, fd)
            self.cross_blocks = nn.ModuleList(
                CrossModalGateFusion(fd, num_heads=cross_attn_heads)
                for _ in range(cross_attn_layers))
            self.out_norm = nn.LayerNorm(fd)
            in_dim = fd                                   # v_t alone
        elif fusion == "legacy_cross_attention":
            self.cross_fusion = CrossModalGateFusion(fd, num_heads=cross_attn_heads)
            in_dim = fd + fd                              # [fused, causal summary]
        else:
            in_dim = fd + fd + td                         # [window mean, causal summary, text]

        self.back_head = _mlp_head(in_dim, hidden_dim)
        self.completion_head = _mlp_head(in_dim, hidden_dim)

        self.freeze_siglip = bool(freeze_siglip)
        if hasattr(self.backbone, "set_siglip_trainable"):
            self.backbone.set_siglip_trainable(not self.freeze_siglip)
        self.cache_text_embeddings = bool(cache_text_embeddings and self.freeze_siglip)
        self._text_cache: dict = {}

    def num_trainable_parameters(self) -> int:
        """Trainable parameters of the trunk and heads (the frozen SigLIP2 towers excluded)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ text encoder
    def _encode_text_uncached(self, texts: List[str], device) -> torch.Tensor:
        tokens = self.backbone.tokenizer(list(texts)).to(device)
        with torch.no_grad():
            emb = self.backbone.siglip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float()

    def encode_skill_text(self, skill_text: SkillText) -> torch.Tensor:
        """Frozen, l2-normalised SigLIP2 text embedding h_theta(l_t) -> [B, text_dim].

        With ``cache_text_embeddings`` each unique skill string is encoded once and reused.
        """
        if isinstance(skill_text, str):
            skill_text = [skill_text]
        device = next(self.parameters()).device
        if not self.cache_text_embeddings:
            return self._encode_text_uncached(skill_text, device)
        missing = [s for s in dict.fromkeys(skill_text) if s not in self._text_cache]
        if missing:
            emb = self._encode_text_uncached(missing, device)
            for s, e in zip(missing, emb, strict=True):
                self._text_cache[s] = e.detach()
        return torch.stack([self._text_cache[s] for s in skill_text], dim=0).to(device)

    # ------------------------------------------------------------------ visual features (f_phi)
    def _visual_outputs_from_siglip(self, feats_tc: torch.Tensor, num_pad_frames: int = 0):
        """Frozen SigLIP2 features ``[B, Tp, Cams, D]`` -> ``(window mean [B, D], frames [B, T, D])``.

        Cameras are averaged per frame; zero-padded frames are dropped so the causal encoder only
        sees real observations."""
        frame_seq = feats_tc.float().mean(dim=2)
        if num_pad_frames:
            frame_seq = frame_seq[:, num_pad_frames:]
        return frame_seq.mean(dim=1), frame_seq

    def _backbone_visual_outputs(self, images: torch.Tensor):
        """Frozen SigLIP2 frame features; always under ``no_grad`` (the towers are frozen)."""
        with torch.no_grad():
            feats_tc, num_pad = self.backbone.encode_siglip_frames(images)
        return self._visual_outputs_from_siglip(feats_tc, num_pad)

    # ------------------------------------------------------------------ trunk g_psi
    def _heads_input(self, outputs, skill_text_emb: torch.Tensor) -> torch.Tensor:
        """Single assembly point for the head input, shared by :meth:`forward` and
        :meth:`forward_from_cached`."""
        pooled, frame_seq = outputs
        pooled, frame_seq = pooled.float(), frame_seq.float()
        txt = skill_text_emb.float()
        if txt.shape[0] == 1 and frame_seq.shape[0] > 1:
            txt = txt.expand(frame_seq.shape[0], -1)

        if self.fusion == "cross_attention":
            temporal = self.temporal_encoder(frame_seq, return_sequence=True)   # [B, T, D]
            q = self.text_proj(txt).unsqueeze(1)                                # [B, 1, D]
            for block in self.cross_blocks:
                q = block(q, temporal)
            return self.out_norm(q.squeeze(1))                                  # v_t [B, D]

        summary = self.temporal_encoder(frame_seq)                              # [B, D]
        if self.fusion == "legacy_cross_attention":
            context = torch.cat([pooled.unsqueeze(1), frame_seq, summary.unsqueeze(1)], dim=1)
            fused = self.cross_fusion(txt.unsqueeze(1), context).squeeze(1)
            return torch.cat([fused, summary], dim=-1)
        return torch.cat([pooled, summary, txt], dim=-1)

    @property
    def cacheable_stage(self):
        """``"frames"`` when the SigLIP2 towers are frozen: per-frame features can be cached once
        and re-gathered into history windows every epoch; ``None`` when they train."""
        return "frames" if self.freeze_siglip else None

    @torch.no_grad()
    def extract_cacheable_features(self, images: torch.Tensor):
        """Frozen-stage outputs to cache plus the (constant) pad-frame count. Only valid when
        :pyattr:`cacheable_stage` is not None."""
        feats_tc, num_pad = self.backbone.encode_siglip_frames(images)
        return (feats_tc,), num_pad

    def forward_from_cached(self, cached_outputs, current_skill_text: SkillText,
                            num_pad_frames: int = 0
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trunk + heads on precomputed SigLIP2 frame features -> ``(completion_logit, back_logit)``."""
        (siglip_feats,) = cached_outputs
        outputs = self._visual_outputs_from_siglip(siglip_feats, num_pad_frames)
        x = self._heads_input(outputs, self.encode_skill_text(current_skill_text))
        return self.completion_head(x).squeeze(-1), self.back_head(x).squeeze(-1)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        images: torch.Tensor,
        current_skill_text: SkillText,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``images`` [B, M, Cams, C, H, W] -> ``(completion_logit[B], back_logit[B])``."""
        outputs = self._backbone_visual_outputs(images)
        x = self._heads_input(outputs, self.encode_skill_text(current_skill_text))
        return self.completion_head(x).squeeze(-1), self.back_head(x).squeeze(-1)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, current_skill_text: SkillText
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: ``(gamma_t, beta_t)`` = (completion, recovery) probabilities in [0, 1]."""
        was_training = self.training
        self.eval()
        completion_logit, back_logit = self.forward(images, current_skill_text)
        if was_training:
            self.train()
        return torch.sigmoid(completion_logit), torch.sigmoid(back_logit)
