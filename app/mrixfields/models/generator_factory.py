"""Generator construction and condition-aware forwarding helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .conditioned_unetr import (
    ConditionedUNETR2DGenerator,
    FIELD_TO_ID,
    MODALITY_TO_ID,
)
# Import the exact hierarchical resize-convolution Swin implementation.
# This avoids depending on a compatibility alias in conditioned_swin_unetr.py.
from .conditioned_swin_unetr_resizeconv import (
    ConditionedSwinUNETR2DGenerator,
)
from .networks import ResnetGenerator, init_net


CONDITIONED_UNETR_NAMES = {
    "conditioned_unetr2d",
    "conditioned_unetr",
    "unetr2d",
    "unetr",
}

CONDITIONED_SWIN_UNETR_NAMES = {
    # Original names.
    "conditioned_swin_unetr2d",
    "conditioned_swin_unetr",
    "swin_unetr2d",
    "swin_unetr",
    "swinunetr",
    # Explicit names for the hierarchical resize-convolution implementation.
    "conditioned_swin_unetr2d_resizeconv",
    "conditioned_swin_unetr_resizeconv",
    "conditioned_swin_resizeconv",
    "swin_unetr2d_resizeconv",
    "swin_unetr_resizeconv",
    "swin_resizeconv",
}


def get_generator_name(model_cfg: dict) -> str:
    """Return the configured generator identifier.

    Accepted configuration keys are checked in priority order so the factory
    remains compatible with older and newer YAML files.
    """
    for key in ("name", "generator", "type", "model_type"):
        value = model_cfg.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return "resnet"


def is_conditioned_generator(model: nn.Module) -> bool:
    """Return whether a generator expects MRI condition identifiers."""
    if hasattr(model, "module"):
        model = model.module
    return bool(getattr(model, "is_conditioned_generator", False))


def build_generator(
    model_cfg: dict,
    data_cfg: Optional[dict] = None,
    task: int = 1,
) -> nn.Module:
    """Build the configured generator while preserving ResNet defaults."""
    data_cfg = data_cfg or {}
    name = get_generator_name(model_cfg)

    if name in {"resnet", "resnet_generator", "cut_resnet"}:
        return ResnetGenerator(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ngf=model_cfg.get("ngf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
        )

    if name in CONDITIONED_UNETR_NAMES:
        modalities = data_cfg.get("modalities") or ["T1W"]
        return ConditionedUNETR2DGenerator(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            base_image_size=model_cfg.get("base_image_size", [256, 256]),
            patch_size=model_cfg.get("patch_size", 16),
            embed_dim=model_cfg.get("embed_dim", 256),
            depth=model_cfg.get("depth", 12),
            num_heads=model_cfg.get("num_heads", 8),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            condition_dim=model_cfg.get("condition_dim", 256),
            decoder_channels=model_cfg.get(
                "decoder_channels", [256, 128, 64, 32]
            ),
            skip_block_indices=model_cfg.get(
                "skip_block_indices", [2, 5, 8, 11]
            ),
            dropout=model_cfg.get("dropout", 0.0),
            attention_dropout=model_cfg.get(
                "attention_dropout", 0.0
            ),
            residual_prediction=model_cfg.get(
                "residual_prediction", True
            ),
            gradient_checkpointing=model_cfg.get(
                "gradient_checkpointing", False
            ),
            default_source_field=data_cfg.get(
                "source_field", "0.1T"
            ),
            default_target_field=data_cfg.get("target_field", "7T"),
            default_modality=modalities[0],
            default_task=task,
        )

    if name in CONDITIONED_SWIN_UNETR_NAMES:
        modalities = data_cfg.get("modalities") or ["T1W"]

        depths = model_cfg.get("depths", [2, 2, 2, 2])
        num_heads = model_cfg.get("num_heads", [3, 6, 12, 24])

        if len(depths) != 4:
            raise ValueError(
                "Conditioned Swin-UNETR requires four depth values, "
                f"received {depths!r}."
            )
        if len(num_heads) != 4:
            raise ValueError(
                "Conditioned Swin-UNETR requires four num_heads values, "
                f"received {num_heads!r}."
            )

        return ConditionedSwinUNETR2DGenerator(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            base_image_size=model_cfg.get(
                "base_image_size",
                data_cfg.get("crop_size", [384, 448]),
            ),
            embed_dim=model_cfg.get("embed_dim", 48),
            depths=depths,
            num_heads=num_heads,
            window_size=model_cfg.get("window_size", 4),
            stem_channels=model_cfg.get("stem_channels", 32),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            condition_dim=model_cfg.get("condition_dim", 192),
            dropout=model_cfg.get("dropout", 0.0),
            attention_dropout=model_cfg.get(
                "attention_dropout", 0.0
            ),
            residual_prediction=model_cfg.get(
                "residual_prediction", True
            ),
            residual_output_mode=model_cfg.get(
                "residual_output_mode", "hard_clamp"
            ),
            residual_scale=model_cfg.get("residual_scale", 1.0),
            gradient_checkpointing=model_cfg.get(
                "gradient_checkpointing", False
            ),
            default_source_field=data_cfg.get(
                "source_field", "0.1T"
            ),
            default_target_field=data_cfg.get("target_field", "7T"),
            default_modality=modalities[0],
            default_task=task,
        )

    supported = sorted(
        {
            "resnet",
            *CONDITIONED_UNETR_NAMES,
            *CONDITIONED_SWIN_UNETR_NAMES,
        }
    )
    raise ValueError(
        f"Unknown generator name {name!r}. "
        f"Supported values include: {', '.join(supported)}."
    )


def initialize_generator(
    generator: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Initialize a generator without destroying AdaLN-Zero initialization."""
    if is_conditioned_generator(generator):
        return generator
    return init_net(generator, init_type, init_gain)


def _batch_id(
    name: str,
    mapping: dict,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if name not in mapping:
        raise ValueError(
            f"Unknown condition {name!r}; expected one of {list(mapping)}."
        )
    return torch.full(
        (batch_size,),
        mapping[name],
        dtype=torch.long,
        device=device,
    )


def forward_generator(
    generator: nn.Module,
    image: torch.Tensor,
    source_field: Optional[str] = None,
    target_field: Optional[str] = None,
    modality: Optional[str] = None,
    task: int = 1,
    layers=None,
    encode_only: bool = False,
):
    """Forward ResNet, conditioned UNETR, or conditioned Swin-UNETR."""
    requested_layers = layers or []

    if not is_conditioned_generator(generator):
        return generator(image, requested_layers, encode_only)

    if source_field is None or target_field is None or modality is None:
        # Constructor defaults remain valid for standard translation calls.
        return generator(image, requested_layers, encode_only)

    batch_size = image.shape[0]
    return generator(
        image,
        requested_layers,
        encode_only,
        source_field_id=_batch_id(
            source_field,
            FIELD_TO_ID,
            batch_size,
            image.device,
        ),
        target_field_id=_batch_id(
            target_field,
            FIELD_TO_ID,
            batch_size,
            image.device,
        ),
        modality_id=_batch_id(
            modality,
            MODALITY_TO_ID,
            batch_size,
            image.device,
        ),
        task_id=torch.full(
            (batch_size,),
            int(task) - 1,
            dtype=torch.long,
            device=image.device,
        ),
    )
