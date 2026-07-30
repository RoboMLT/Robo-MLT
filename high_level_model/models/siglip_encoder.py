"""Frozen SigLIP2 image + text towers for System 2 (f_phi / h_theta, paper Eq. 3).

This is the entire "backbone" the trunk-free :class:`CompletionGate` needs: a per-frame
appearance feature (via ``encode_siglip_frames``) and a skill-text embedding (via
``siglip_model.encode_text`` / ``tokenizer``, called directly by ``CompletionGate``). It
carries no trunk, no fusion transformer, and no closed-set retrieval head — those lived in
the legacy ``TaskConditionedInstructor`` but are not part of the published method.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import open_clip
import torchvision.transforms as transforms

__all__ = ["SigLIPFrameEncoder", "build_siglip_encoder"]


class SigLIPFrameEncoder(nn.Module):
    """Frozen SigLIP2 image + text towers (f_phi / h_theta in Eq. 3).

    Exposes exactly the surface :class:`~high_level_model.models.completion_gate.CompletionGate`
    touches: ``siglip_model`` (with ``encode_text`` / ``encode_image``), ``tokenizer``,
    ``vision_output_dim``, ``set_siglip_trainable`` and ``encode_siglip_frames``.
    """

    def __init__(
        self,
        device: str = "cuda",
        history_len: int = 6,
        model_name: str = "ViT-B-16-SigLIP2",
        pretrained: str = "webli",
        freeze_siglip: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.history_len = history_len
        self.freeze_siglip = freeze_siglip

        self.siglip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.vision_output_dim = self.siglip_model.visual.trunk.embed_dim
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.tensor_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.set_siglip_trainable(not freeze_siglip)

    def set_siglip_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the SigLIP tower; ``encode_siglip_frames`` reads ``freeze_siglip``
        to decide whether to run under ``no_grad``."""
        self.freeze_siglip = not trainable
        for p in self.siglip_model.parameters():
            p.requires_grad = trainable

    def encode_siglip_frames(self, images: torch.Tensor):
        """Per-frame SigLIP appearance features.

        ``images`` ``[B, T, Cams, C, H, W]`` -> ``(feats [B, Tp, Cams, D], num_pad_frames)``,
        where ``Tp = max(T, history_len + 1)`` (left zero-padded when the history is short).
        This is the only frozen computation in the model, so it is what gets cached across
        epochs when ``freeze_siglip`` is True.
        """
        batch_size, timesteps, num_cameras, c, h, w = images.shape

        num_pad_frames = 0
        if timesteps < self.history_len + 1:
            num_pad_frames = self.history_len + 1 - timesteps
            padding = torch.zeros(
                (batch_size, num_pad_frames, num_cameras, c, h, w), device=images.device
            )
            images = torch.cat([padding, images], dim=1)
            timesteps = self.history_len + 1

        images_reshaped = images.reshape(batch_size * timesteps * num_cameras, c, h, w)
        images_transformed = self.tensor_transform(images_reshaped)
        vision_ctx = torch.no_grad() if self.freeze_siglip else nullcontext()
        with vision_ctx:
            image_features = self.siglip_model.encode_image(images_transformed)

        image_features_tc = image_features.reshape(
            batch_size, timesteps, num_cameras, -1
        ).float()
        return image_features_tc, num_pad_frames


def build_siglip_encoder(
    device: str = "cuda",
    history_len: int = 6,
    freeze_siglip: bool = True,
    model_name: str = "ViT-B-16-SigLIP2",
    pretrained: str = "webli",
) -> SigLIPFrameEncoder:
    """Convenience constructor mirroring the old ``build_instructor_for_lerobot`` call sites,
    minus the dataset dependency: no ``repo_id`` is needed because the trunk-free gate never
    uses a closed-set candidate list."""
    return SigLIPFrameEncoder(
        device=device,
        history_len=history_len,
        model_name=model_name,
        pretrained=pretrained,
        freeze_siglip=freeze_siglip,
    )
