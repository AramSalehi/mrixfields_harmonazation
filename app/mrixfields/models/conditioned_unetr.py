"""Conditioned 2D UNETR generator for MRI field-strength translation.

The generator is CUT-compatible:
- normal forward returns a translated image in [-1, 1]
- ``encode_only=True`` returns spatial encoder features for PatchNCE
- source field, target field, modality, and task conditions are injected into
  every ViT block with AdaLN-Zero and into every decoder level with FiLM
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FIELD_STRENGTHS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]


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


def _modulate_tokens(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MRIConditionEncoder(nn.Module):
    """Embed source field, target field, modality, and task metadata."""

    def __init__(
        self,
        condition_dim: int,
        field_embedding_dim: int = 64,
        modality_embedding_dim: int = 32,
        task_embedding_dim: int = 32,
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
        self.task_embedding = nn.Embedding(3, task_embedding_dim)

        total_dim = (
            2 * field_embedding_dim
            + modality_embedding_dim
            + task_embedding_dim
        )
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
            self.task_embedding,
        ):
            nn.init.normal_(embedding.weight, std=0.02)

        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        values = [
            self.source_field_embedding(source_field_id),
            self.target_field_embedding(target_field_id),
            self.modality_embedding(modality_id),
            self.task_embedding(task_id),
        ]
        return self.projection(torch.cat(values, dim=-1))


class ConditionedViTBlock(nn.Module):
    """Pre-norm ViT block with condition-dependent AdaLN-Zero."""

    def __init__(
        self,
        embed_dim: int,
        condition_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by "
                f"num_heads={num_heads}."
            )

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.condition_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * embed_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # AdaLN-Zero keeps each residual branch initially close to zero.
        modulation = self.condition_modulation[-1]
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.condition_modulation(condition).chunk(6, dim=-1)

        attention_input = _modulate_tokens(
            self.norm1(tokens), shift_attn, scale_attn
        )
        attention_output, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        tokens = tokens + gate_attn.unsqueeze(1) * attention_output

        mlp_input = _modulate_tokens(
            self.norm2(tokens), shift_mlp, scale_mlp
        )
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return tokens


class ConditionalResidualConvBlock(nn.Module):
    """Residual convolution block with FiLM at both normalization layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        groups = _largest_group_count(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.condition_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 4 * out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(
                self.skip.weight, mode="fan_out", nonlinearity="relu"
            )
            if self.skip.bias is not None:
                nn.init.zeros_(self.skip.bias)

        projection = self.condition_projection[-1]
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)

    @staticmethod
    def _film(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return (
            x * (1.0 + scale[:, :, None, None])
            + shift[:, :, None, None]
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.condition_projection(
            condition
        ).chunk(4, dim=-1)

        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self._film(x, shift1, scale1)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self._film(x, shift2, scale2)
        return self.activation(x + residual)


class ConditionedDecoderStage(nn.Module):
    """Upsample, concatenate a skip, and refine with conditional FiLM."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        condition_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.block = ConditionalResidualConvBlock(
            out_channels + skip_channels,
            out_channels,
            condition_dim,
            dropout,
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
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.block(torch.cat([x, skip], dim=1), condition)


class TokenProjection(nn.Module):
    """Convert [B, N, C] tokens to a projected [B, C_out, H, W] map."""

    def __init__(self, embed_dim: int, out_channels: int) -> None:
        super().__init__()
        groups = _largest_group_count(out_channels)
        self.projection = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        grid_h, grid_w = grid_size
        if token_count != grid_h * grid_w:
            raise ValueError(
                f"Token count {token_count} does not match "
                f"grid {grid_h}x{grid_w}."
            )
        x = tokens.transpose(1, 2).reshape(
            batch_size, channels, grid_h, grid_w
        )
        return self.projection(x)


class ConditionedUNETR2DGenerator(nn.Module):
    """CUT-compatible 2D UNETR generator with pervasive conditioning."""

    is_conditioned_generator = True

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        base_image_size: Sequence[int] = (256, 256),
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        condition_dim: int = 256,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        skip_block_indices: Sequence[int] = (2, 5, 8, 11),
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_prediction: bool = True,
        gradient_checkpointing: bool = False,
        default_source_field: str = "0.1T",
        default_target_field: str = "7T",
        default_modality: str = "T1W",
        default_task: int = 1,
    ) -> None:
        super().__init__()

        if patch_size != 16:
            raise ValueError(
                "The current UNETR decoder uses four 2x stages and "
                "therefore requires patch_size=16."
            )
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four values.")
        if len(skip_block_indices) != 4:
            raise ValueError("skip_block_indices must contain four values.")
        if max(skip_block_indices) >= depth or min(skip_block_indices) < 0:
            raise ValueError(
                "skip_block_indices must be valid transformer block indices."
            )
        if sorted(skip_block_indices) != list(skip_block_indices):
            raise ValueError("skip_block_indices must be in ascending order.")

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.base_image_size = _as_pair(base_image_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.skip_block_indices = tuple(int(i) for i in skip_block_indices)
        self.residual_prediction = residual_prediction
        self.gradient_checkpointing = gradient_checkpointing

        self.default_source_field_id = FIELD_TO_ID[default_source_field]
        self.default_target_field_id = FIELD_TO_ID[default_target_field]
        self.default_modality_id = MODALITY_TO_ID[default_modality]
        self.default_task_id = int(default_task) - 1

        base_h = (self.base_image_size[0] + patch_size - 1) // patch_size
        base_w = (self.base_image_size[1] + patch_size - 1) // patch_size
        self.base_grid_size = (base_h, base_w)

        self.condition_encoder = MRIConditionEncoder(
            condition_dim=condition_dim,
            dropout=dropout,
        )
        self.patch_embedding = nn.Conv2d(
            input_nc,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, base_h * base_w, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [
                ConditionedViTBlock(
                    embed_dim=embed_dim,
                    condition_dim=condition_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        c4, c3, c2, c1 = [int(value) for value in decoder_channels]
        self.skip1_projection = TokenProjection(embed_dim, c1)
        self.skip2_projection = TokenProjection(embed_dim, c2)
        self.skip3_projection = TokenProjection(embed_dim, c3)
        self.skip4_projection = TokenProjection(embed_dim, c4)

        self.skip1_refine = ConditionalResidualConvBlock(
            c1, c1, condition_dim, dropout
        )
        self.skip2_refine = ConditionalResidualConvBlock(
            c2, c2, condition_dim, dropout
        )
        self.skip3_refine = ConditionalResidualConvBlock(
            c3, c3, condition_dim, dropout
        )
        self.skip4_refine = ConditionalResidualConvBlock(
            c4, c4, condition_dim, dropout
        )

        self.skip1_up = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, 2, 2),
            nn.SiLU(),
            nn.ConvTranspose2d(c1, c1, 2, 2),
            nn.SiLU(),
            nn.ConvTranspose2d(c1, c1, 2, 2),
            nn.SiLU(),
        )
        self.skip2_up = nn.Sequential(
            nn.ConvTranspose2d(c2, c2, 2, 2),
            nn.SiLU(),
            nn.ConvTranspose2d(c2, c2, 2, 2),
            nn.SiLU(),
        )
        self.skip3_up = nn.Sequential(
            nn.ConvTranspose2d(c3, c3, 2, 2),
            nn.SiLU(),
        )

        self.bottleneck = ConditionalResidualConvBlock(
            c4, c4, condition_dim, dropout
        )
        self.decoder3 = ConditionedDecoderStage(
            c4, c3, c3, condition_dim, dropout
        )
        self.decoder2 = ConditionedDecoderStage(
            c3, c2, c2, condition_dim, dropout
        )
        self.decoder1 = ConditionedDecoderStage(
            c2, c1, c1, condition_dim, dropout
        )
        self.input_stem = ConditionalResidualConvBlock(
            input_nc, c1, condition_dim, dropout
        )
        self.decoder0 = ConditionedDecoderStage(
            c1, c1, c1, condition_dim, dropout
        )
        self.final_refinement = ConditionalResidualConvBlock(
            c1, c1, condition_dim, dropout
        )
        self.output_head = nn.Conv2d(c1, output_nc, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.patch_embedding.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.output_head.weight)
        if self.output_head.bias is not None:
            nn.init.zeros_(self.output_head.bias)

    @staticmethod
    def _batch_ids(
        value: Optional[torch.Tensor | int],
        default_value: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(
                (batch_size,), default_value, dtype=torch.long, device=device
            )
        if isinstance(value, int):
            return torch.full(
                (batch_size,), value, dtype=torch.long, device=device
            )
        value = value.to(device=device, dtype=torch.long)
        if value.ndim == 0:
            value = value.expand(batch_size)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"Condition batch has length {value.shape[0]}, "
                f"but image batch has length {batch_size}."
            )
        return value

    def _condition(
        self,
        image: torch.Tensor,
        source_field_id: Optional[torch.Tensor | int],
        target_field_id: Optional[torch.Tensor | int],
        modality_id: Optional[torch.Tensor | int],
        task_id: Optional[torch.Tensor | int],
    ) -> torch.Tensor:
        batch_size = image.shape[0]
        device = image.device
        return self.condition_encoder(
            self._batch_ids(
                source_field_id,
                self.default_source_field_id,
                batch_size,
                device,
            ),
            self._batch_ids(
                target_field_id,
                self.default_target_field_id,
                batch_size,
                device,
            ),
            self._batch_ids(
                modality_id,
                self.default_modality_id,
                batch_size,
                device,
            ),
            self._batch_ids(
                task_id,
                self.default_task_id,
                batch_size,
                device,
            ),
        )

    def _pad_to_patch_multiple(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        original_h, original_w = image.shape[-2:]
        pad_h = (-original_h) % self.patch_size
        pad_w = (-original_w) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return image, (original_h, original_w)

        mode = "reflect"
        if original_h <= pad_h or original_w <= pad_w:
            mode = "replicate"
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
        return padded, (original_h, original_w)

    def _position_for_grid(
        self,
        grid_size: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        base_h, base_w = self.base_grid_size
        position = self.position_embedding.reshape(
            1, base_h, base_w, self.embed_dim
        ).permute(0, 3, 1, 2)
        if grid_size != self.base_grid_size:
            position = F.interpolate(
                position,
                size=grid_size,
                mode="bicubic",
                align_corners=False,
            )
        return position.flatten(2).transpose(1, 2).to(
            device=device, dtype=dtype
        )

    @staticmethod
    def _tokens_to_map(
        tokens: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        grid_h, grid_w = grid_size
        if token_count != grid_h * grid_w:
            raise ValueError(
                f"Token count {token_count} does not match "
                f"grid {grid_h}x{grid_w}."
            )
        return tokens.transpose(1, 2).reshape(
            batch_size, channels, grid_h, grid_w
        )

    def _run_block(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            try:
                return checkpoint(
                    block,
                    tokens,
                    condition,
                    use_reentrant=False,
                )
            except TypeError:
                return checkpoint(block, tokens, condition)
        return block(tokens, condition)

    def _encode(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        requested_indices: Iterable[int],
    ) -> Tuple[Dict[int, torch.Tensor], Tuple[int, int]]:
        features = self.patch_embedding(image)
        grid_size = tuple(features.shape[-2:])
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self._position_for_grid(
            grid_size, tokens.dtype, tokens.device
        )
        tokens = self.embedding_dropout(tokens)

        requested = set(int(index) for index in requested_indices)
        outputs: Dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.transformer_blocks):
            tokens = self._run_block(block, tokens, condition)
            if index in requested:
                outputs[index] = tokens
        return outputs, grid_size

    def _decode(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        token_features: Dict[int, torch.Tensor],
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        index1, index2, index3, index4 = self.skip_block_indices
        token1 = token_features[index1]
        token2 = token_features[index2]
        token3 = token_features[index3]
        token4 = token_features[index4]

        skip1 = self.skip1_projection(token1, grid_size)
        skip1 = self.skip1_refine(skip1, condition)
        skip1 = self.skip1_up(skip1)

        skip2 = self.skip2_projection(token2, grid_size)
        skip2 = self.skip2_refine(skip2, condition)
        skip2 = self.skip2_up(skip2)

        skip3 = self.skip3_projection(token3, grid_size)
        skip3 = self.skip3_refine(skip3, condition)
        skip3 = self.skip3_up(skip3)

        skip4 = self.skip4_projection(token4, grid_size)
        skip4 = self.skip4_refine(skip4, condition)

        x = self.bottleneck(skip4, condition)
        x = self.decoder3(x, skip3, condition)
        x = self.decoder2(x, skip2, condition)
        x = self.decoder1(x, skip1, condition)
        x = self.decoder0(x, self.input_stem(image, condition), condition)
        x = self.final_refinement(x, condition)
        residual_or_image = self.output_head(x)

        if self.residual_prediction:
            if self.output_nc != 1:
                raise ValueError(
                    "Residual prediction currently requires output_nc=1."
                )
            center_channel = self.input_nc // 2
            reference = image[:, center_channel:center_channel + 1]
            # The zero-initialized output head starts as an exact identity map.
            return torch.clamp(reference + residual_or_image, -1.0, 1.0)

        # Match the existing GAN pipeline, whose images are in [-1, 1].
        return torch.tanh(residual_or_image)

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
        layers = [self.depth - 1 if index == -1 else int(index) for index in layers]
        for index in layers:
            if index < 0 or index >= self.depth:
                raise ValueError(
                    f"Requested NCE layer {index}, but depth={self.depth}."
                )

        padded, original_size = self._pad_to_patch_multiple(input)
        condition = self._condition(
            padded,
            source_field_id,
            target_field_id,
            modality_id,
            task_id,
        )

        required_indices = set(layers)
        if not encode_only:
            required_indices.update(self.skip_block_indices)

        token_features, grid_size = self._encode(
            padded, condition, required_indices
        )

        requested_features = [
            self._tokens_to_map(token_features[index], grid_size)
            for index in layers
        ]
        if encode_only:
            return requested_features

        output = self._decode(
            padded, condition, token_features, grid_size
        )
        output = output[..., :original_size[0], :original_size[1]]

        if layers:
            return output, requested_features
        return output
