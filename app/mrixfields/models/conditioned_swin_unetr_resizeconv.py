"""Conditioned 2D Swin-UNETR generator for MRI field-strength translation.

Drop-in replacement for ``ConditionedUNETR2DGenerator`` with the same
CUT-compatible interface:

    forward(input, layers=[], encode_only=False,
            source_field_id=None, target_field_id=None,
            modality_id=None, task_id=None)

    * normal call            -> translated image in [-1, 1]
    * ``layers=[...]``       -> (image, [feature maps]) for PatchNCE
    * ``encode_only=True``   -> [feature maps] only
    * ``is_conditioned_generator = True``
    * residual head is zero-initialised, so the model starts as an exact
      identity map (output == source image) and learns a field residual.

Two deliberate departures from the plain UNETR generator:

  1. A hierarchical Swin encoder (patch_size=2, four stages with patch
     merging) replaces the single stride-16 ViT grid. Features exist at
     strides 1/2/4/8/16, so there is no coarse patch lattice to imprint the
     "quilt" blocking artifact.

  2. Every upsample is resize-conv -- ``F.interpolate`` (bilinear) followed by
     a 3x3 ``Conv2d``. There is no ``ConvTranspose2d`` anywhere in this module,
     which removes the transposed-convolution checkerboard.

Conditioning (source field, target field, *numerical* field delta, and modality)
is injected into every Swin block via AdaLN-Zero and into every decoder conv block via FiLM.
The task identifier is accepted by the public API for backward compatibility but is not encoded.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FIELD_STRENGTHS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELD_TESLA = [0.1, 1.5, 3.0, 5.0, 7.0]

FIELD_TO_ID = {name: index for index, name in enumerate(FIELD_STRENGTHS)}
MODALITY_TO_ID = {name: index for index, name in enumerate(MODALITIES)}


def _as_pair(value: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a pair, received {value!r}.")
    return int(value[0]), int(value[1])


def _largest_group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


# --------------------------------------------------------------------------- #
#  Conditioning
# --------------------------------------------------------------------------- #

class MRIConditionEncoder(nn.Module):
    """Embed source field, target field, numerical field delta, and modality.

    The numerical delta (target_tesla - source_tesla) gives the network an
    ordered, magnitude-aware prior: 0.1T->7T is a bigger jump than 5T->7T.
    Categorical embeddings still capture field-specific characteristics.
    """

    def __init__(
        self,
        condition_dim: int,
        field_embedding_dim: int = 64,
        modality_embedding_dim: int = 32,
        task_embedding_dim: int = 32,
        delta_embedding_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.source_field_embedding = nn.Embedding(
            len(FIELD_STRENGTHS), field_embedding_dim
        )
        self.target_field_embedding = nn.Embedding(
            len(FIELD_STRENGTHS), field_embedding_dim
        )
        self.modality_embedding = nn.Embedding(
            len(MODALITIES), modality_embedding_dim
        )
        self.field_delta_mlp = nn.Sequential(
            nn.Linear(1, delta_embedding_dim),
            nn.SiLU(),
            nn.Linear(delta_embedding_dim, delta_embedding_dim),
        )

        self.register_buffer(
            "field_tesla",
            torch.tensor(FIELD_TESLA, dtype=torch.float32),
            persistent=False,
        )

        total_dim = (
            2 * field_embedding_dim
            + modality_embedding_dim
            + delta_embedding_dim
        )
        # Concatenate-then-project keeps the five signals separable rather than
        # aliasing them into a single summed vector.
        self.projection = nn.Sequential(
            nn.Linear(total_dim, condition_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.source_field_embedding,
            self.target_field_embedding,
            self.modality_embedding,
        ):
            nn.init.normal_(embedding.weight, std=0.02)
        for module in list(self.projection) + list(self.field_delta_mlp):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tesla = self.field_tesla
        delta = (
            tesla[target_field_id] - tesla[source_field_id]
        ).unsqueeze(-1)
        values = [
            self.source_field_embedding(source_field_id),
            self.target_field_embedding(target_field_id),
            self.modality_embedding(modality_id),
            self.field_delta_mlp(delta),
        ]
        return self.projection(torch.cat(values, dim=-1))


# --------------------------------------------------------------------------- #
#  Window attention (Swin)
# --------------------------------------------------------------------------- #

def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window * window, c
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window: int, h: int, w: int
) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window / window))
    x = windows.view(b, h // window, w // window, window, window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention within local windows, with relative bias."""

    def __init__(
        self,
        dim: int,
        window: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window = window
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) * (2 * window - 1), num_heads)
        )
        coords_h = torch.arange(window)
        coords_w = torch.arange(window)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window - 1
        relative[:, :, 1] += window - 1
        relative[:, :, 0] *= 2 * window - 1
        self.register_buffer(
            "relative_position_index", relative.sum(-1), persistent=False
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(
            b_, n, 3, self.num_heads, c // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window * self.window, self.window * self.window, -1
        )
        bias = bias.permute(2, 0, 1).contiguous()
        attention = attention + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                b_ // num_windows, num_windows, self.num_heads, n, n
            ) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, n, n)

        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        x = (attention @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.projection(x)
        x = self.projection_dropout(x)
        return x


class ConditionedSwinBlock(nn.Module):
    """Swin transformer block with AdaLN-Zero conditioning.

    Uses non-affine LayerNorm; the shift/scale/gate come from the condition
    vector. Gates are zero-initialised so the block starts as identity.
    """

    def __init__(
        self,
        dim: int,
        condition_dim: int,
        num_heads: int,
        window: int,
        shift: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window = window
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = WindowAttention(
            dim, window, num_heads, attention_dropout, dropout
        )
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.condition_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * dim),
        )
        nn.init.zeros_(self.condition_modulation[-1].weight)
        nn.init.zeros_(self.condition_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        h: int,
        w: int,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, n, c = x.shape
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.condition_modulation(condition).chunk(6, dim=-1)

        shortcut = x
        y = self.norm1(x)
        y = y * (1.0 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        y = y.view(b, h, w, c)

        if self.shift > 0:
            y = torch.roll(y, shifts=(-self.shift, -self.shift), dims=(1, 2))
            attn_mask = mask
        else:
            attn_mask = None

        windows = window_partition(y, self.window)
        windows = self.attention(windows, attn_mask)
        y = window_reverse(windows, self.window, h, w)

        if self.shift > 0:
            y = torch.roll(y, shifts=(self.shift, self.shift), dims=(1, 2))
        y = y.view(b, n, c)
        x = shortcut + gate_attn.unsqueeze(1) * y

        z = self.norm2(x)
        z = z * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(z)
        return x


class PatchMerging(nn.Module):
    """2x spatial downsample; channels double."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    """A Swin stage: alternating W-MSA / SW-MSA blocks, then optional merge."""

    def __init__(
        self,
        dim: int,
        depth: int,
        condition_dim: int,
        num_heads: int,
        window: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.window = window
        self.blocks = nn.ModuleList(
            [
                ConditionedSwinBlock(
                    dim=dim,
                    condition_dim=condition_dim,
                    num_heads=num_heads,
                    window=window,
                    shift=0 if (i % 2 == 0) else window // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def _attention_mask(
        self, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        shift = self.window // 2
        img_mask = torch.zeros((1, h, w, 1), device=device)
        slices = (
            slice(0, -self.window),
            slice(-self.window, -shift),
            slice(-shift, None),
        )
        count = 0
        for hs in slices:
            for ws in slices:
                img_mask[:, hs, ws, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window).squeeze(-1)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float(-100.0))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, h: int, w: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        mask = self._attention_mask(h, w, x.device)
        for block in self.blocks:
            x = block(x, condition, h, w, mask)
        skip = x
        skip_hw = (h, w)
        if self.downsample is not None:
            x = self.downsample(x, h, w)
            h, w = h // 2, w // 2
        return x, skip, skip_hw, (h, w)


# --------------------------------------------------------------------------- #
#  Convolutional blocks + resize-conv upsampling (no ConvTranspose2d)
# --------------------------------------------------------------------------- #

class ConditionalResidualConvBlock(nn.Module):
    """Residual conv block with FiLM at both GroupNorm layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        groups = _largest_group_count(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels, affine=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, affine=False)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.condition_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 4 * out_channels),
        )
        nn.init.zeros_(self.condition_projection[-1].weight)
        nn.init.zeros_(self.condition_projection[-1].bias)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.condition_projection(
            condition
        ).chunk(4, dim=-1)
        shift1 = shift1[..., None, None]
        scale1 = scale1[..., None, None]
        shift2 = shift2[..., None, None]
        scale2 = scale2[..., None, None]

        residual = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y) * (1.0 + scale1) + shift1
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y) * (1.0 + scale2) + shift2
        y = self.activation(y)
        return residual + y


class ResizeConvUp(nn.Module):
    """2x upsample via bilinear interpolation + 3x3 conv (no ConvTranspose2d)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        return self.conv(x)


class ConditionedDecoderStage(nn.Module):
    """Resize-conv upsample, concatenate a skip, refine with conditional FiLM."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        condition_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = ResizeConvUp(in_channels, out_channels)
        self.block = ConditionalResidualConvBlock(
            out_channels + skip_channels, out_channels, condition_dim, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.block(torch.cat([x, skip], dim=1), condition)


# --------------------------------------------------------------------------- #
#  Generator
# --------------------------------------------------------------------------- #

class ConditionedSwinUNETR2DGenerator(nn.Module):
    """CUT-compatible 2D Swin-UNETR generator with pervasive field/modality conditioning."""

    is_conditioned_generator = True

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        base_image_size: Sequence[int] = (256, 256),
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 8,
        stem_channels: int = 48,
        mlp_ratio: float = 4.0,
        condition_dim: int = 256,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_prediction: bool = True,
        residual_output_mode: str = "hard_clamp",
        residual_scale: float = 1.0,
        gradient_checkpointing: bool = False,
        default_source_field: str = "0.1T",
        default_target_field: str = "7T",
        default_modality: str = "T1W",
        default_task: int = 1,
        **_ignored,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("depths and num_heads must each have 4 entries.")

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.base_image_size = _as_pair(base_image_size)
        self.window_size = window_size
        self.residual_prediction = residual_prediction
        self.residual_output_mode = str(residual_output_mode).strip().lower()
        self.residual_scale = float(residual_scale)
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be > 0")
        self.gradient_checkpointing = gradient_checkpointing
        # patch_size(2) * 2**(num_stages-1) * window_size keeps every stage's
        # resolution divisible by the window, so no per-stage padding is needed.
        self.size_divisor = 2 * (2 ** 3) * window_size

        self.default_source_field_id = FIELD_TO_ID[default_source_field]
        self.default_target_field_id = FIELD_TO_ID[default_target_field]
        self.default_modality_id = MODALITY_TO_ID[default_modality]
        self.default_task_id = int(default_task) - 1

        self.condition_encoder = MRIConditionEncoder(
            condition_dim=condition_dim, dropout=dropout
        )

        # Stem at full resolution -> genuine stride-1 skip.
        self.input_stem = ConditionalResidualConvBlock(
            input_nc, stem_channels, condition_dim, dropout
        )

        # Swin patch embed: stride 2.
        self.patch_embedding = nn.Conv2d(
            input_nc, embed_dim, kernel_size=2, stride=2
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        dims = [embed_dim * (2 ** i) for i in range(4)]  # 96,192,384,768
        self.stages = nn.ModuleList(
            [
                BasicLayer(
                    dim=dims[i],
                    depth=depths[i],
                    condition_dim=condition_dim,
                    num_heads=num_heads[i],
                    window=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    downsample=(i < 3),
                )
                for i in range(4)
            ]
        )

        # Skip refinement (stride 2/4/8) + bottleneck (stride 16).
        self.skip1_refine = ConditionalResidualConvBlock(
            dims[0], dims[0], condition_dim, dropout
        )
        self.skip2_refine = ConditionalResidualConvBlock(
            dims[1], dims[1], condition_dim, dropout
        )
        self.skip3_refine = ConditionalResidualConvBlock(
            dims[2], dims[2], condition_dim, dropout
        )
        self.bottleneck = ConditionalResidualConvBlock(
            dims[3], dims[3], condition_dim, dropout
        )

        # Decoder: 16 -> 8 -> 4 -> 2 -> 1, all resize-conv.
        self.decoder3 = ConditionedDecoderStage(
            dims[3], dims[2], dims[2], condition_dim, dropout
        )
        self.decoder2 = ConditionedDecoderStage(
            dims[2], dims[1], dims[1], condition_dim, dropout
        )
        self.decoder1 = ConditionedDecoderStage(
            dims[1], dims[0], dims[0], condition_dim, dropout
        )
        self.decoder0 = ConditionedDecoderStage(
            dims[0], stem_channels, stem_channels, condition_dim, dropout
        )
        self.final_refinement = ConditionalResidualConvBlock(
            stem_channels, stem_channels, condition_dim, dropout
        )
        self.output_head = nn.Conv2d(stem_channels, output_nc, kernel_size=1)

        # Zero-init head: residual model starts as an exact identity map.
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # Feature maps exposed to PatchNCE, coarse -> fine ordering handled below.
        self._encode_dims = [stem_channels, dims[0], dims[1], dims[2], dims[3]]
        self.num_feature_layers = len(self._encode_dims)

    # --- conditioning helpers ------------------------------------------- #

    @staticmethod
    def _batch_ids(
        value: Optional[torch.Tensor | int],
        default: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(
                (batch_size,), default, dtype=torch.long, device=device
            )
        if isinstance(value, int):
            return torch.full(
                (batch_size,), value, dtype=torch.long, device=device
            )
        return value.to(device=device, dtype=torch.long).reshape(batch_size)

    def _condition(
        self,
        image: torch.Tensor,
        source_field_id,
        target_field_id,
        modality_id,
        task_id,
    ) -> torch.Tensor:
        b = image.shape[0]
        d = image.device
        return self.condition_encoder(
            self._batch_ids(source_field_id, self.default_source_field_id, b, d),
            self._batch_ids(target_field_id, self.default_target_field_id, b, d),
            self._batch_ids(modality_id, self.default_modality_id, b, d),
            self._batch_ids(task_id, self.default_task_id, b, d),
        )

    def _pad(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = image.shape[-2:]
        pad_h = (-h) % self.size_divisor
        pad_w = (-w) % self.size_divisor
        if pad_h == 0 and pad_w == 0:
            return image, (h, w)
        mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"
        return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), (h, w)

    # --- encoder / decoder ---------------------------------------------- #

    def _run_stage(self, stage, x, condition, h, w):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(stage, x, condition, h, w, use_reentrant=False)
        return stage(x, condition, h, w)

    def _encode(self, image, condition):
        x = self.patch_embedding(image)
        _, _, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.embedding_dropout(self.patch_norm(x))

        skips = []          # (tokens, (h, w)) per stage, fine -> coarse
        h, w = gh, gw
        for stage in self.stages:
            x, skip, skip_hw, (h, w) = self._run_stage(
                stage, x, condition, h, w
            )
            skips.append((skip, skip_hw))
        return skips

    @staticmethod
    def _to_map(tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        h, w = hw
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def _decode(self, image, condition, skips):
        (s1, hw1), (s2, hw2), (s3, hw3), (s4, hw4) = skips
        skip1 = self.skip1_refine(self._to_map(s1, hw1), condition)  # stride 2
        skip2 = self.skip2_refine(self._to_map(s2, hw2), condition)  # stride 4
        skip3 = self.skip3_refine(self._to_map(s3, hw3), condition)  # stride 8
        bottleneck = self.bottleneck(self._to_map(s4, hw4), condition)  # s16

        stem = self.input_stem(image, condition)  # stride 1

        x = self.decoder3(bottleneck, skip3, condition)
        x = self.decoder2(x, skip2, condition)
        x = self.decoder1(x, skip1, condition)
        x = self.decoder0(x, stem, condition)
        x = self.final_refinement(x, condition)
        out = self.output_head(x)

        if self.residual_prediction:
            if self.output_nc != 1:
                raise ValueError(
                    "Residual prediction currently requires output_nc=1."
                )
            center = self.input_nc // 2
            reference = image[:, center:center + 1]

            # Stable residual output.  The historical implementation used a
            # hard clamp(reference + out, -1, 1), which can create a large
            # zero-gradient region after saturation.  The atanh/tanh mode
            # keeps the output bounded while remaining differentiable.  With
            # the zero-initialized output head it starts as (approximately)
            # an identity mapping, preserving the intended residual design.
            if self.residual_output_mode in {"atanh_tanh", "stable", "stable_residual"}:
                eps = 1e-4
                reference_safe = reference.clamp(-1.0 + eps, 1.0 - eps)
                return torch.tanh(
                    torch.atanh(reference_safe) + self.residual_scale * out
                )
            if self.residual_output_mode in {"hard_clamp", "clamp", "legacy"}:
                return torch.clamp(
                    reference + self.residual_scale * out, -1.0, 1.0
                )
            raise ValueError(
                f"Unknown residual_output_mode={self.residual_output_mode!r}. "
                "Use 'atanh_tanh' or 'hard_clamp'."
            )
        return torch.tanh(out)

    def _encoder_feature_maps(self, image, condition):
        """Multi-scale maps for PatchNCE: [stem(s1), s2, s4, s8, s16]."""
        skips = self._encode(image, condition)
        stem = self.input_stem(image, condition)
        maps = [stem] + [self._to_map(s, hw) for (s, hw) in skips]
        return maps

    # --- public forward -------------------------------------------------- #

    def forward(
        self,
        input: torch.Tensor,
        layers: Optional[Sequence[int]] = None,
        encode_only: bool = False,
        source_field_id: Optional[torch.Tensor | int] = None,
        target_field_id: Optional[torch.Tensor | int] = None,
        modality_id: Optional[torch.Tensor | int] = None,
        task_id: Optional[torch.Tensor | int] = None,
    ):
        layers = list(layers or [])
        n = self.num_feature_layers
        layers = [n - 1 if i == -1 else int(i) for i in layers]
        for i in layers:
            if i < 0 or i >= n:
                raise ValueError(
                    f"Requested feature layer {i}, but only {n} are available."
                )

        padded, original_size = self._pad(input)
        condition = self._condition(
            padded, source_field_id, target_field_id, modality_id, task_id
        )

        if encode_only:
            maps = self._encoder_feature_maps(padded, condition)
            return [maps[i] for i in layers]

        skips = self._encode(padded, condition)
        output = self._decode(padded, condition, skips)
        output = output[..., :original_size[0], :original_size[1]]

        if layers:
            stem = self.input_stem(padded, condition)
            maps = [stem] + [self._to_map(s, hw) for (s, hw) in skips]
            return output, [maps[i] for i in layers]
        return output

# Optional compatibility alias for code that expects the older generator class name.
ConditionedUNETR2DGenerator = ConditionedSwinUNETR2DGenerator
