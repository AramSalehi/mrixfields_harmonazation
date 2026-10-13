"""Training loops for the Task 3 conditioned-UNETR CUT method."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from audit_utils import get_mem_usage, get_train_logger
from audit_dataset import audit_worker_init_fn, audited_torch_load

from .data.cached_dataset import CachedPairedDataset, cached_subject_id
from .data.dataset import PairedMRIDataset
from .data.task3_dataset import (
    CachedTask3UnpairedDataset,
    ConditionedPairedDataset,
    Task3UnpairedMRIDataset,
)
from .data.utils import ABBR_TO_SPLIT as SPLIT_MAP
from .losses.adversarial import GANLoss
from .losses.perceptual import PerceptualLoss
from .losses.structure import SSIMLoss, StructureLoss
from .models.conditioned_unetr import FIELD_TO_ID, MODALITY_TO_ID
from .models.generator_factory import build_generator, initialize_generator
from .models.task3_cut_model import (
    ConditionalPatchDiscriminator,
    Task3ConditionedCUTModel,
    condition_ids,
)
from .utils_dist import get_dist_info, is_main_process
from .wandb_logging import (
    image_payload,
    log_wandb,
    should_log,
    should_log_images,
    update_summary,
)


AUDIT_LOGGER = get_train_logger(console=False)




def _runtime_max_iters(train_cfg: dict, loader_len: int) -> int:
    """Resolve an optional per-epoch data-fraction cap for smoke tests.

    MRI_XFIELDS_DATA_FRACTION=0.01 means at most ceil(1% of the available
    training batches) are consumed per epoch. An explicit max_iters_per_epoch
    remains an additional upper bound. Full runs default to 1.0 and are
    unchanged.
    """
    explicit = int(train_cfg.get("max_iters_per_epoch", 0) or 0)
    raw = os.environ.get("MRI_XFIELDS_DATA_FRACTION", "1.0")
    try:
        fraction = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"MRI_XFIELDS_DATA_FRACTION must be numeric, got {raw!r}"
        ) from exc
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "MRI_XFIELDS_DATA_FRACTION must be > 0 and <= 1.0"
        )

    fraction_cap = 0
    if fraction < 1.0:
        fraction_cap = max(1, int(math.ceil(loader_len * fraction)))

    if explicit and fraction_cap:
        return min(explicit, fraction_cap)
    if explicit:
        return min(explicit, loader_len)
    return fraction_cap

def _audited_loader_kwargs(
    num_workers: int,
    *,
    pin_memory: bool,
) -> dict:
    """Return DataLoader options with one raw audit log per forked worker."""
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs.update(
            {
                "worker_init_fn": audit_worker_init_fn,
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )

    return kwargs


def _audit(message: str) -> None:
    """Write an official audit training record from rank 0 only."""
    if is_main_process():
        AUDIT_LOGGER.info(message)


@contextmanager
def _preserve_training_rng_state():
    """Prevent audit validation from changing the stochastic train path.

    CUT/PatchNCE samples patches with PyTorch RNG. Validation is required for
    the audit, but the original pre-audit trainer had no validation pass. This
    context saves and restores Python, NumPy, CPU-Torch, and CUDA RNG states so
    validation cannot perturb the random sequence used by subsequent training.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _prediction_diagnostics(prefix: str, tensor: torch.Tensor) -> dict:
    """Non-random W&B diagnostics for detecting output collapse."""
    x = tensor.detach()
    finite = torch.isfinite(x)
    safe = torch.where(finite, x, torch.zeros_like(x))
    return {
        f"{prefix}/min": float(safe.min()),
        f"{prefix}/max": float(safe.max()),
        f"{prefix}/mean": float(safe.mean()),
        f"{prefix}/std": float(safe.std(unbiased=False)),
        f"{prefix}/finite_fraction": float(finite.float().mean()),
        f"{prefix}/at_minus_one_fraction": float((safe <= -0.999).float().mean()),
        f"{prefix}/at_plus_one_fraction": float((safe >= 0.999).float().mean()),
    }


def _validation_config(cfg: dict) -> dict:
    raw = dict(cfg.get("validation", {}))
    fraction = float(raw.get("fraction", 0.1))
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation.fraction must be between 0 and 1.")
    return {
        "fraction": fraction,
        "seed": int(raw.get("seed", 2026)),
        "batch_size": int(raw.get("batch_size", 8)),
        "max_batches": int(raw.get("max_batches", 25)),
        "stage1_start_after_epoch": int(
            raw.get("stage1_start_after_epoch", 40)
        ),
    }


def _split_subjects(
    subjects: Sequence[str], fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    unique = sorted(set(subjects))
    if len(unique) < 2:
        raise ValueError(
            "Audited subject-disjoint validation requires at least two subjects."
        )
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n_validation = max(1, int(round(len(shuffled) * fraction)))
    n_validation = min(n_validation, len(shuffled) - 1)
    validation = sorted(shuffled[:n_validation])
    training = sorted(shuffled[n_validation:])
    return training, validation


def _write_data_split(
    output_dir: Path,
    *,
    stage: str,
    source_split: str,
    training_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    seed: int,
    fraction: float,
    split_method: str = "deterministic subject-level holdout",
    validation_overlaps_training: bool = False,
) -> None:
    if not is_main_process():
        return
    record = {
        "stage": stage,
        "source_split": source_split,
        "split_method": split_method,
        "validation_overlaps_training": bool(validation_overlaps_training),
        "seed": seed,
        "validation_fraction": fraction,
        "training_subjects": list(training_subjects),
        "validation_subjects": list(validation_subjects),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_split.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {stage} data split",
        "",
        f"- Source split: `{source_split}`",
        f"- Method: {split_method}",
        f"- Validation overlaps training: `{validation_overlaps_training}`",
        f"- Seed: `{seed}`",
        f"- Requested validation fraction: `{fraction}`",
        f"- Training subjects: `{len(training_subjects)}`",
        f"- Validation subjects: `{len(validation_subjects)}`",
        "",
        "## Training subject list",
        "",
        *[f"- {subject}" for subject in training_subjects],
        "",
        "## Validation subject list",
        "",
        *[f"- {subject}" for subject in validation_subjects],
        "",
    ]
    (output_dir / "data-split.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _audit(
        "[DataSplit] "
        f"Stage:{stage}, SourceSplit:{source_split}, Seed:{seed}, "
        f"SplitMethod:{split_method}, "
        f"ValidationOverlapsTraining:{validation_overlaps_training}, "
        f"ValidationFraction:{fraction}, "
        f"TrainSubjects:{list(training_subjects)}, "
        f"ValidationSubjects:{list(validation_subjects)}"
    )


def _log_validation(
    *, stage: str, epoch: int, global_step: int, values: dict, event: str
) -> None:
    _audit(
        f"[Validation] Stage:{stage}, Event:{event}, Epoch:{epoch}, "
        f"GlobalStep:{global_step}, ValTotalLoss:{values['loss_total']:.8f}, "
        f"ValLosses:{json.dumps(values, sort_keys=True)}"
    )


def _log_checkpoint(
    *,
    stage: str,
    path: Path,
    epoch: int,
    global_step: int,
    validation: Optional[dict] = None,
    is_best: Optional[bool] = None,
) -> None:
    """Log checkpoint creation without forcing validation into training.

    In the isolated-validation pipeline, training writes the checkpoint first.
    A separate audited process later evaluates that immutable checkpoint and
    emits a linked [CheckpointValidation] record.
    """
    if validation is None:
        _audit(
            f"[Checkpoint] Stage:{stage}, SavedTo:{path.resolve()}, "
            f"Epoch:{epoch}, GlobalStep:{global_step}, "
            "ValidationStatus:pending_external_process"
        )
        return

    _audit(
        f"[Checkpoint] Stage:{stage}, SavedTo:{path.resolve()}, "
        f"Epoch:{epoch}, GlobalStep:{global_step}, IsBest:{is_best}, "
        f"ValLoss:{validation['loss_total']:.8f}, "
        f"ValMetrics:{json.dumps(validation, sort_keys=True)}"
    )


def _lr_lambda(n_epochs: int, n_epochs_decay: int):
    def rule(epoch: int) -> float:
        return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
    return rule


# Explicit configuration aliases used by the resize-convolution Swin model.
# Task3ConditionedCUTModel currently recognizes the canonical
# "conditioned_swin_unetr" name, so normalize aliases before constructing it.
_SWIN_RESIZECONV_ALIASES = {
    "conditioned_swin_unetr2d_resizeconv",
    "conditioned_swin_unetr_resizeconv",
    "conditioned_swin_resizeconv",
    "swin_unetr2d_resizeconv",
    "swin_unetr_resizeconv",
    "swin_resizeconv",
}

_SWIN_CANONICAL_NAMES = {
    "conditioned_swin_unetr",
    "conditioned_swin_unetr2d",
    "swin_unetr",
    "swin_unetr2d",
    "swinunetr",
}


def _normalize_task3_model_config(model_cfg: dict) -> dict:
    """Normalize Task 3 Swin aliases and validate PatchNCE feature layers.

    The hierarchical resize-convolution Swin generator exposes five CUT
    feature maps with valid indices 0, 1, 2, 3, and 4.
    """
    normalized = dict(model_cfg)

    raw_name = None
    for key in ("name", "generator", "type", "model_type"):
        value = normalized.get(key)
        if value is not None and str(value).strip():
            raw_name = str(value).strip().lower()
            break
    name = raw_name or "resnet"

    if name in _SWIN_RESIZECONV_ALIASES:
        name = "conditioned_swin_unetr"

    if name in _SWIN_CANONICAL_NAMES:
        # Use one canonical identifier so task3_cut_model.py and the generator
        # factory cannot disagree about the configured architecture.
        normalized["name"] = "conditioned_swin_unetr"
        normalized["generator"] = "conditioned_swin_unetr"
        normalized["type"] = "conditioned_swin_unetr"

        configured_layers = normalized.get("nce_layers")
        try:
            layers = (
                [int(layer) for layer in configured_layers]
                if configured_layers is not None
                else [0, 1, 2, 3, 4]
            )
        except (TypeError, ValueError):
            layers = [0, 1, 2, 3, 4]

        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(layer < 0 or layer > 4 for layer in layers)
        ):
            layers = [0, 1, 2, 3, 4]

        normalized["nce_layers"] = layers

    return normalized


def _extract_generator_state(state: dict) -> dict:
    if "model" in state:
        for prefix in ("netG.module.", "netG."):
            result = {
                key[len(prefix):]: value
                for key, value in state["model"].items()
                if key.startswith(prefix)
            }
            if result:
                return result
        raise ValueError("Task 3 checkpoint contains no netG generator weights.")
    if "generator" in state:
        return state["generator"]
    return state


def _extract_discriminator_state(state: dict) -> dict:
    """Extract the conditional PatchGAN state from Stage-1 or Stage-2 checkpoints."""
    if "model" in state:
        for prefix in ("netD.module.", "netD."):
            result = {
                key[len(prefix):]: value
                for key, value in state["model"].items()
                if key.startswith(prefix)
            }
            if result:
                return result
        raise ValueError("Task 3 checkpoint contains no netD discriminator weights.")
    if "discriminator" in state:
        return state["discriminator"]
    raise ValueError(
        "Conditional Stage-2 GAN requested, but checkpoint contains no discriminator state."
    )


def _conditions(batch: Dict, device: torch.device):
    source_field_id = condition_ids(
        batch["source_field"], FIELD_TO_ID, device
    )
    target_field_id = condition_ids(
        batch["target_field"], FIELD_TO_ID, device
    )
    modality_id = condition_ids(
        batch["modality"], MODALITY_TO_ID, device
    )
    return source_field_id, target_field_id, modality_id


def _finetune_augmentation_config(finetune_cfg: dict) -> dict:
    """Return validated paired fine-tuning augmentation settings."""
    raw = finetune_cfg.get("augmentation", {})
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    elif raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raise TypeError(
            "finetune.augmentation must be a mapping or boolean, "
            f"got {type(raw).__name__}."
        )

    cfg = {
        "enabled": bool(raw.get("enabled", False)),
        # Synchronized spatial transforms: always applied identically to source
        # and paired target so voxel correspondence is preserved.
        "hflip_prob": float(raw.get("hflip_prob", 0.0)),
        "vflip_prob": float(raw.get("vflip_prob", 0.0)),
        "rotate_prob": float(raw.get("rotate_prob", 0.0)),
        "rotate_degrees": float(raw.get("rotate_degrees", 0.0)),
        # Mild synchronized elastic deformation. alpha is the approximate
        # maximum displacement in pixels; grid_size controls smoothness.
        "elastic_prob": float(raw.get("elastic_prob", 0.0)),
        "elastic_alpha": float(raw.get("elastic_alpha", 1.5)),
        "elastic_grid_size": int(raw.get("elastic_grid_size", 8)),
        # Mild intensity perturbation. By default this is applied only to the
        # source image so the target remains the true target-field appearance.
        "intensity_prob": float(raw.get("intensity_prob", 0.0)),
        "gamma_min": float(raw.get("gamma_min", 1.0)),
        "gamma_max": float(raw.get("gamma_max", 1.0)),
        "scale_min": float(raw.get("scale_min", 1.0)),
        "scale_max": float(raw.get("scale_max", 1.0)),
        "shift": float(raw.get("shift", 0.0)),
        "noise_std": float(raw.get("noise_std", 0.0)),
        "source_only_intensity": bool(
            raw.get("source_only_intensity", True)
        ),
    }

    for key in (
        "hflip_prob",
        "vflip_prob",
        "rotate_prob",
        "elastic_prob",
        "intensity_prob",
    ):
        if not 0.0 <= cfg[key] <= 1.0:
            raise ValueError(f"finetune.augmentation.{key} must be in [0, 1].")

    if cfg["rotate_degrees"] < 0:
        raise ValueError(
            "finetune.augmentation.rotate_degrees must be non-negative."
        )
    if cfg["elastic_alpha"] < 0:
        raise ValueError(
            "finetune.augmentation.elastic_alpha must be non-negative."
        )
    if cfg["elastic_grid_size"] < 2:
        raise ValueError(
            "finetune.augmentation.elastic_grid_size must be at least 2."
        )
    if cfg["gamma_min"] <= 0 or cfg["gamma_max"] <= 0:
        raise ValueError(
            "finetune.augmentation gamma limits must be positive."
        )
    if cfg["gamma_min"] > cfg["gamma_max"]:
        raise ValueError(
            "finetune.augmentation.gamma_min cannot exceed gamma_max."
        )
    if cfg["scale_min"] <= 0 or cfg["scale_max"] <= 0:
        raise ValueError(
            "finetune.augmentation scale limits must be positive."
        )
    if cfg["scale_min"] > cfg["scale_max"]:
        raise ValueError(
            "finetune.augmentation.scale_min cannot exceed scale_max."
        )
    if cfg["shift"] < 0 or cfg["noise_std"] < 0:
        raise ValueError(
            "finetune.augmentation shift and noise_std must be non-negative."
        )
    return cfg


def _masked_batch_choice(
    original: torch.Tensor,
    transformed: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Choose the transformed tensor independently for each batch item."""
    if probability <= 0:
        return original
    batch_mask = (
        torch.rand(original.shape[0], device=original.device) < probability
    ).view(-1, 1, 1, 1)
    return torch.where(batch_mask, transformed, original)


def _apply_intensity_transform(
    image: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    """Apply a mild transform to normalized [-1, 1] images.

    Voxels at the background floor are preserved to avoid creating a bright
    border around the skull-stripped brain.
    """
    foreground = image > -0.999
    image_01 = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
    transformed = image_01.pow(gamma)
    transformed = transformed * scale + shift
    if noise_std > 0:
        transformed = transformed + torch.randn_like(transformed) * noise_std
    transformed = transformed.clamp(0.0, 1.0) * 2.0 - 1.0
    return torch.where(foreground, transformed, image)


def _elastic_deformation_grid(
    image: torch.Tensor,
    probability: float,
    alpha_pixels: float,
    control_grid_size: int,
) -> torch.Tensor | None:
    """Create a smooth per-sample elastic sampling grid.

    The same returned grid must be applied to source and target to preserve
    paired correspondence. ``alpha_pixels`` is an approximate maximum
    displacement in image pixels.
    """
    if probability <= 0 or alpha_pixels <= 0:
        return None

    batch_size, _, height, width = image.shape
    device = image.device
    dtype = image.dtype

    apply_mask = (
        torch.rand(batch_size, device=device) < probability
    ).to(dtype).view(batch_size, 1, 1, 1)

    # Low-resolution random displacement is upsampled to obtain a smooth field.
    displacement = torch.empty(
        batch_size,
        2,
        control_grid_size,
        control_grid_size,
        device=device,
        dtype=dtype,
    ).uniform_(-1.0, 1.0)
    displacement = F.interpolate(
        displacement,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    displacement = displacement.clamp(-1.0, 1.0) * apply_mask

    # Convert pixel displacement to grid_sample normalized coordinates.
    displacement[:, 0] *= 2.0 * alpha_pixels / max(width, 1)
    displacement[:, 1] *= 2.0 * alpha_pixels / max(height, 1)

    y = torch.linspace(
        -1.0 + 1.0 / height,
        1.0 - 1.0 / height,
        height,
        device=device,
        dtype=dtype,
    )
    x = torch.linspace(
        -1.0 + 1.0 / width,
        1.0 - 1.0 / width,
        width,
        device=device,
        dtype=dtype,
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    base_grid = base_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return base_grid + displacement.permute(0, 2, 3, 1)


def _augment_paired_finetune_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    aug_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply optional online augmentation to a paired fine-tuning batch.

    Geometric operations are synchronized between source and target.
    Intensity augmentation is source-only by default.
    """
    if not aug_cfg["enabled"]:
        return source, target

    # Left-right flip.
    if aug_cfg["hflip_prob"] > 0:
        mask = (
            torch.rand(source.shape[0], device=source.device)
            < aug_cfg["hflip_prob"]
        ).view(-1, 1, 1, 1)
        source = torch.where(mask, torch.flip(source, dims=(-1,)), source)
        target = torch.where(mask, torch.flip(target, dims=(-1,)), target)

    # Superior-inferior image-axis flip. Kept disabled by default for axial MRI.
    if aug_cfg["vflip_prob"] > 0:
        mask = (
            torch.rand(source.shape[0], device=source.device)
            < aug_cfg["vflip_prob"]
        ).view(-1, 1, 1, 1)
        source = torch.where(mask, torch.flip(source, dims=(-2,)), source)
        target = torch.where(mask, torch.flip(target, dims=(-2,)), target)

    # Small synchronized in-plane rotation while retaining the original H x W.
    if aug_cfg["rotate_prob"] > 0 and aug_cfg["rotate_degrees"] > 0:
        batch_size = source.shape[0]
        apply_rotation = (
            torch.rand(batch_size, device=source.device)
            < aug_cfg["rotate_prob"]
        )
        max_radians = aug_cfg["rotate_degrees"] * torch.pi / 180.0
        angles = (
            torch.rand(batch_size, device=source.device) * 2.0 - 1.0
        ) * max_radians
        angles = torch.where(apply_rotation, angles, torch.zeros_like(angles))

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        theta = torch.zeros(
            batch_size,
            2,
            3,
            device=source.device,
            dtype=source.dtype,
        )
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        grid = F.affine_grid(theta, source.shape, align_corners=False)
        source = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        target = F.grid_sample(
            target,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    # Mild synchronized smooth elastic deformation.
    elastic_grid = _elastic_deformation_grid(
        source,
        probability=aug_cfg["elastic_prob"],
        alpha_pixels=aug_cfg["elastic_alpha"],
        control_grid_size=aug_cfg["elastic_grid_size"],
    )
    if elastic_grid is not None:
        source = F.grid_sample(
            source,
            elastic_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        target = F.grid_sample(
            target,
            elastic_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    # Mild photometric perturbation.
    if aug_cfg["intensity_prob"] > 0:
        batch_size = source.shape[0]
        apply_intensity = (
            torch.rand(batch_size, device=source.device)
            < aug_cfg["intensity_prob"]
        ).view(-1, 1, 1, 1)

        dtype = source.dtype
        device = source.device
        gamma = torch.empty(
            batch_size, 1, 1, 1, device=device, dtype=dtype
        ).uniform_(aug_cfg["gamma_min"], aug_cfg["gamma_max"])
        scale = torch.empty(
            batch_size, 1, 1, 1, device=device, dtype=dtype
        ).uniform_(aug_cfg["scale_min"], aug_cfg["scale_max"])
        shift = torch.empty(
            batch_size, 1, 1, 1, device=device, dtype=dtype
        ).uniform_(-aug_cfg["shift"], aug_cfg["shift"])

        source_aug = _apply_intensity_transform(
            source,
            gamma=gamma,
            scale=scale,
            shift=shift,
            noise_std=aug_cfg["noise_std"],
        )
        source = torch.where(apply_intensity, source_aug, source)

        if not aug_cfg["source_only_intensity"]:
            target_aug = _apply_intensity_transform(
                target,
                gamma=gamma,
                scale=scale,
                shift=shift,
                noise_std=0.0,
            )
            target = torch.where(apply_intensity, target_aug, target)

    return source, target


def _validate_task3_cut(
    model: Task3ConditionedCUTModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict:
    with _preserve_training_rng_state():
        was_training = model.training
        model.eval()
        sums = {
            "loss_total": 0.0,
            "loss_G": 0.0,
            "loss_D": 0.0,
            "loss_G_GAN": 0.0,
            "loss_NCE": 0.0,
            "loss_identity": 0.0,
        }
        completed = 0
        with torch.no_grad():
            for iteration, batch in enumerate(loader):
                if max_batches and iteration >= max_batches:
                    break
                source = batch["source"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                source_ids, target_ids, modality_ids = _conditions(batch, device)
                model.forward(
                    source, target, source_ids, target_ids, modality_ids
                )
                loss_d = model.compute_D_loss(
                    target, target_ids, modality_ids
                )
                loss_g = model.compute_G_loss(
                    source, target, source_ids, target_ids, modality_ids
                )
                values = {
                    "loss_G": float(loss_g.detach()),
                    "loss_D": float(loss_d.detach()),
                    "loss_G_GAN": float(model.loss_G_GAN),
                    "loss_NCE": float(model.loss_NCE),
                    "loss_identity": float(model.loss_identity),
                }
                values["loss_total"] = values["loss_G"] + values["loss_D"]
                for key in sums:
                    sums[key] += values[key]
                completed += 1
        if was_training:
            model.train()
        return {key: value / max(completed, 1) for key, value in sums.items()}


def train_task3_conditioned_cut(
    cfg: dict,
    device: torch.device,
    use_dist: bool = False,
    resume_ckpt: str | None = None,
) -> None:
    """Audited unpaired retrospective pretraining for Task 3."""
    if use_dist:
        raise NotImplementedError(
            "The final audited pipeline is intentionally single-GPU so that "
            "official per-file logs are not interleaved across worker ranks."
        )

    model_cfg = _normalize_task3_model_config(cfg["model"])
    cfg["model"] = model_cfg
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    validation_cfg = _validation_config(cfg)

    output_dir = Path(train_cfg["output_dir"])
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MODEL] Task 3 generator: {model_cfg.get('name')}")
        print(f"[MODEL] PatchNCE layers: {model_cfg.get('nce_layers')}")

    model = Task3ConditionedCUTModel(
        model_config=model_cfg,
        data_config=data_cfg,
        lr=train_cfg.get("lr", 2e-4),
        discriminator_lr=train_cfg.get("discriminator_lr"),
        gradient_clip_norm=train_cfg.get("gradient_clip_norm", 0.0),
        beta1=train_cfg.get("beta1", 0.5),
        beta2=train_cfg.get("beta2", 0.999),
    ).to(device)
    model.init_weights()

    modalities = list(data_cfg.get("modalities", ["T1W", "T2W", "T2FLAIR"]))
    fields = list(data_cfg.get("field_strengths", ["0.1T", "1.5T", "3T", "5T", "7T"]))
    crop_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_raw) if crop_raw else None
    split = train_cfg.get("split", "retro_train")
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    if not preprocessed_dir:
        raise RuntimeError(
            "The audited pipeline requires PREPROCESSED_DIR so that explicit "
            "subject-level train/validation lists can be recorded."
        )

    inventory = CachedTask3UnpairedDataset(
        preprocessed_dir=preprocessed_dir,
        split=split,
        modalities=modalities,
        field_strengths=fields,
        crop_size=crop_size,
        deterministic=True,
        deterministic_seed=validation_cfg["seed"],
    )
    all_subjects = sorted(inventory.subject_ids)
    _, validation_subjects = _split_subjects(
        all_subjects,
        validation_cfg["fraction"],
        validation_cfg["seed"],
    )

    # Match the known-working pre-audit behavior: every retrospective subject
    # remains eligible for Stage 1 training. The deterministic subset exists
    # only to provide the required per-epoch audit validation loss.
    training_subjects = all_subjects
    training_dataset = CachedTask3UnpairedDataset(
        preprocessed_dir=preprocessed_dir,
        split=split,
        modalities=modalities,
        field_strengths=fields,
        crop_size=crop_size,
        deterministic=False,
    )
    # Construct the deterministic validation loader lazily at the first
    # configured validation epoch. For this experiment the config uses zero
    # warm-up epochs, so validation runs after epoch 1 and epoch 2.
    validation_loader = None
    stage1_validation_start_after_epoch = int(
        validation_cfg["stage1_start_after_epoch"]
    )
    _write_data_split(
        output_dir,
        stage="retro_scratch",
        source_split=split,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        seed=validation_cfg["seed"],
        fraction=validation_cfg["fraction"],
        split_method=(
            "full retrospective training inventory + deterministic audit "
            "monitoring subset"
        ),
        validation_overlaps_training=True,
    )

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_options = _audited_loader_kwargs(
        num_workers,
        pin_memory=device.type == "cuda",
    )
    loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Full-epoch experiment: keep the final incomplete batch so every
        # retrospective source slice is visited once per epoch.
        drop_last=False,
        **loader_options,
    )
    init_batch = next(iter(loader))
    init_source = init_batch["source"].to(device)
    source_ids, target_ids, modality_ids = _conditions(init_batch, device)
    model.data_dependent_initialize(
        init_source, source_ids, target_ids, modality_ids
    )
    model.setup_optimizers()

    n_epochs = int(train_cfg.get("n_epochs", 100))
    n_epochs_decay = int(train_cfg.get("n_epochs_decay", 0))
    rule = _lr_lambda(n_epochs, n_epochs_decay)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(model.optimizer_G, rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(model.optimizer_D, rule)
    scheduler_F = torch.optim.lr_scheduler.LambdaLR(model.optimizer_F, rule)

    start_epoch = 0
    best_validation = float("inf")
    if resume_ckpt:
        state = audited_torch_load(resume_ckpt, torch, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer_G" in state:
            model.optimizer_G.load_state_dict(state["optimizer_G"])
            model.optimizer_D.load_state_dict(state["optimizer_D"])
            model.optimizer_F.load_state_dict(state["optimizer_F"])
            scheduler_G.load_state_dict(state["scheduler_G"])
            scheduler_D.load_state_dict(state["scheduler_D"])
            scheduler_F.load_state_dict(state["scheduler_F"])
        start_epoch = int(state.get("epoch", 0))
        best_validation = float(
            state.get("best_validation", float("inf"))
        )

    total_epochs = n_epochs + n_epochs_decay
    max_iters = _runtime_max_iters(train_cfg, len(loader))
    steps_per_epoch = min(len(loader), max_iters) if max_iters else len(loader)
    global_step = start_epoch * max(steps_per_epoch, 1)

    # Stability guard for the black-image collapse observed in the previous
    # long Stage-1 run.  This does not alter the loss; it monitors generator
    # saturation, keeps a rolling iteration checkpoint, and aborts early if
    # the output remains almost entirely at the lower intensity bound.
    collapse_check_every = int(train_cfg.get("collapse_check_every", 100) or 0)
    collapse_threshold = float(
        train_cfg.get("collapse_minus_one_threshold", 0.95)
    )
    collapse_patience = int(train_cfg.get("collapse_patience", 3) or 1)
    collapse_abort = bool(train_cfg.get("collapse_abort", True))
    iteration_checkpoint_every = int(
        train_cfg.get("iteration_checkpoint_every", 2000) or 0
    )
    collapse_hits = 0

    _audit(
        "[StageStart] Stage:retro_scratch, Initialization:"
        + ("resume" if resume_ckpt else "random")
        + f", EpochStart:{start_epoch}, TotalEpochs:{total_epochs}, "
        f"BatchSize:{batch_size}, StepsPerEpoch:{steps_per_epoch}, "
        f"TrainSubjects:{len(training_subjects)}, "
        f"ValidationSubjects:{len(validation_subjects)}, "
        f"FullDatasetPass:{max_iters == 0}, "
        f"RuntimeDataFraction:{os.environ.get('MRI_XFIELDS_DATA_FRACTION', '1.0')}, "
        f"DropLast:false, "
        f"ValidationExecution:"
        + (
            "epoch_end_after_every_complete_epoch"
            if stage1_validation_start_after_epoch == 0
            else f"none_through_epoch_{stage1_validation_start_after_epoch}_then_epoch_end"
        )
    )

    # Save the initialized model before the first optimizer update. Serialization
    # does not consume training RNG; external validation can therefore evaluate
    # initial performance without executing validation inside this process.
    if is_main_process() and start_epoch == 0 and not resume_ckpt:
        initial_checkpoint_path = weights_dir / "checkpoint_epoch0.pth"
        torch.save(
            {
                "epoch": 0,
                "global_step": 0,
                "model": model.state_dict(),
                "optimizer_G": model.optimizer_G.state_dict(),
                "optimizer_D": model.optimizer_D.state_dict(),
                "optimizer_F": model.optimizer_F.state_dict(),
                "scheduler_G": scheduler_G.state_dict(),
                "scheduler_D": scheduler_D.state_dict(),
                "scheduler_F": scheduler_F.state_dict(),
                "data_split": {
                    "training_subjects": training_subjects,
                    "validation_subjects": validation_subjects,
                },
            },
            initial_checkpoint_path,
        )
        print(f"Saved initial snapshot: {initial_checkpoint_path}")
        _log_checkpoint(
            stage="retro_scratch",
            path=initial_checkpoint_path,
            epoch=0,
            global_step=0,
        )

    for epoch in range(start_epoch, total_epochs):
        model.train()
        sums = {
            "loss_G": 0.0,
            "loss_D": 0.0,
            "loss_G_GAN": 0.0,
            "loss_NCE": 0.0,
            "loss_identity": 0.0,
        }
        completed = 0
        progress = tqdm(
            loader,
            desc=f"Task3 epoch {epoch + 1}/{total_epochs}",
            disable=not is_main_process(),
        )
        for iteration, batch in enumerate(progress):
            if max_iters and iteration >= max_iters:
                break
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            source_ids, target_ids, modality_ids = _conditions(batch, device)
            losses = model.optimize_parameters(
                source, target, source_ids, target_ids, modality_ids
            )
            completed += 1
            global_step += 1
            for key in sums:
                sums[key] += losses[key]

            # Always compute the inexpensive scalar diagnostics at the selected
            # interval, even if W&B image logging is disabled.
            if collapse_check_every and global_step % collapse_check_every == 0:
                diag = _prediction_diagnostics(
                    "task3_prediction_stats", model.fake_target
                )
                minus_frac = diag[
                    "task3_prediction_stats/at_minus_one_fraction"
                ]
                finite_frac = diag[
                    "task3_prediction_stats/finite_fraction"
                ]
                if is_main_process():
                    print(
                        f"[STABILITY] step={global_step} "
                        f"min={diag['task3_prediction_stats/min']:.5f} "
                        f"max={diag['task3_prediction_stats/max']:.5f} "
                        f"mean={diag['task3_prediction_stats/mean']:.5f} "
                        f"std={diag['task3_prediction_stats/std']:.5f} "
                        f"at_minus_one={minus_frac:.6f} "
                        f"finite={finite_frac:.6f}",
                        flush=True,
                    )
                bad = (finite_frac < 1.0) or (minus_frac >= collapse_threshold)
                collapse_hits = collapse_hits + 1 if bad else 0
                if collapse_hits >= collapse_patience:
                    if is_main_process():
                        emergency_path = (
                            weights_dir / f"checkpoint_COLLAPSE_step{global_step}.pth"
                        )
                        torch.save(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "model": model.state_dict(),
                                "optimizer_G": model.optimizer_G.state_dict(),
                                "optimizer_D": model.optimizer_D.state_dict(),
                                "optimizer_F": model.optimizer_F.state_dict(),
                                "scheduler_G": scheduler_G.state_dict(),
                                "scheduler_D": scheduler_D.state_dict(),
                                "scheduler_F": scheduler_F.state_dict(),
                                "stability_diagnostics": diag,
                                "collapse_hits": collapse_hits,
                            },
                            emergency_path,
                        )
                        print(
                            f"[STABILITY] COLLAPSE DETECTED; emergency checkpoint: "
                            f"{emergency_path}",
                            flush=True,
                        )
                        _audit(
                            f"[CollapseDetected] Stage:retro_scratch, "
                            f"Epoch:{epoch + 1}, GlobalStep:{global_step}, "
                            f"AtMinusOneFraction:{minus_frac:.8f}, "
                            f"FiniteFraction:{finite_frac:.8f}, "
                            f"Checkpoint:{emergency_path.resolve()}"
                        )
                    if collapse_abort:
                        raise RuntimeError(
                            "Generator output collapse detected: "
                            f"at_minus_one_fraction={minus_frac:.6f} for "
                            f"{collapse_hits} consecutive checks."
                        )

            # Rolling checkpoint for long diagnostics.  The filename is
            # intentionally reused so a 2-epoch batch-size-2 run does not
            # create hundreds of multi-GB checkpoint files.
            if (
                is_main_process()
                and iteration_checkpoint_every
                and global_step % iteration_checkpoint_every == 0
            ):
                iter_path = weights_dir / "checkpoint_latest_iteration.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model": model.state_dict(),
                        "optimizer_G": model.optimizer_G.state_dict(),
                        "optimizer_D": model.optimizer_D.state_dict(),
                        "optimizer_F": model.optimizer_F.state_dict(),
                        "scheduler_G": scheduler_G.state_dict(),
                        "scheduler_D": scheduler_D.state_dict(),
                        "scheduler_F": scheduler_F.state_dict(),
                    },
                    iter_path,
                )
                print(
                    f"[CHECKPOINT] rolling iteration checkpoint at step "
                    f"{global_step}: {iter_path}",
                    flush=True,
                )

            if is_main_process():
                progress.set_postfix(
                    G=f"{losses['loss_G']:.4f}",
                    D=f"{losses['loss_D']:.4f}",
                    NCE=f"{losses['loss_NCE']:.4f}",
                )
                _audit(
                    f"[Training] Stage:retro_scratch, Epoch:{epoch + 1}, "
                    f"Iteration:{iteration + 1}, GlobalStep:{global_step}, "
                    f"LR_G:{model.optimizer_G.param_groups[0]['lr']}, "
                    f"LR_D:{model.optimizer_D.param_groups[0]['lr']}, "
                    f"LR_F:{model.optimizer_F.param_groups[0]['lr']}, "
                    f"Scheduler:LambdaLR, BatchSize:{source.shape[0]}, "
                    f"LossTotal:{losses['loss_G'] + losses['loss_D']:.8f}, "
                    f"Losses:{json.dumps(losses, sort_keys=True)}, "
                    f"LossWeights:{json.dumps({'GAN': model.lambda_GAN, 'NCE': model.lambda_NCE, 'Identity': model.lambda_identity})}, "
                    f"MemUsage:{get_mem_usage()}"
                )
                if should_log(global_step):
                    payload = {f"task3/{key}": value for key, value in losses.items()}
                    payload.update({
                        "task3/lr_G": model.optimizer_G.param_groups[0]["lr"],
                        "task3/lr_D": model.optimizer_D.param_groups[0]["lr"],
                        "task3/epoch": epoch + 1,
                    })
                    if should_log_images(global_step):
                        payload.update(
                            _prediction_diagnostics(
                                "task3_prediction_stats", model.fake_target
                            )
                        )
                        payload.update(
                            image_payload(
                                source=source,
                                prediction=model.fake_target,
                                target=target,
                                prefix="task3_images",
                            )
                        )
                    log_wandb(payload, step=global_step)

        scheduler_G.step()
        scheduler_D.step()
        scheduler_F.step()

        averages = {
            key: value / max(completed, 1) for key, value in sums.items()
        }
        current_epoch = epoch + 1

        # Validate at epoch end according to stage1_start_after_epoch. The
        # new full2 config sets this to 0, so both complete epochs are validated.
        validation = None
        is_best = False
        if current_epoch > stage1_validation_start_after_epoch:
            if validation_loader is None:
                validation_dataset = CachedTask3UnpairedDataset(
                    preprocessed_dir=preprocessed_dir,
                    split=split,
                    modalities=modalities,
                    field_strengths=fields,
                    crop_size=crop_size,
                    include_subjects=validation_subjects,
                    deterministic=True,
                    deterministic_seed=validation_cfg["seed"],
                )
                validation_loader = DataLoader(
                    validation_dataset,
                    batch_size=validation_cfg["batch_size"],
                    shuffle=False,
                    drop_last=False,
                    **_audited_loader_kwargs(
                        num_workers,
                        pin_memory=device.type == "cuda",
                    ),
                )
                _audit(
                    f"[ValidationStart] Stage:retro_scratch, "
                    f"FirstValidationEpoch:{current_epoch}, "
                    f"WarmupEpochsWithoutValidation:{stage1_validation_start_after_epoch}, "
                    f"ValidationSubjects:{len(validation_subjects)}"
                )

            validation = _validate_task3_cut(
                model,
                validation_loader,
                device,
                validation_cfg["max_batches"],
            )
            _log_validation(
                stage="retro_scratch",
                epoch=current_epoch,
                global_step=global_step,
                values=validation,
                event="epoch_end_after_warmup",
            )
            is_best = validation["loss_total"] < best_validation
            if is_best:
                best_validation = validation["loss_total"]

        if is_main_process():
            message = (
                f"Task3 epoch {current_epoch}: "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in averages.items()
                )
            )
            if validation is not None:
                message += f" val_total={validation['loss_total']:.6f}"
            print(message)

            wandb_payload = {
                **{
                    f"task3_epoch/{key}": value
                    for key, value in averages.items()
                },
                "task3_epoch/index": current_epoch,
            }
            if validation is not None:
                wandb_payload.update({
                    f"task3_validation/{key}": value
                    for key, value in validation.items()
                })
            log_wandb(wandb_payload, step=global_step)

            summary_payload = {
                "last_task3_epoch": current_epoch,
                **{
                    f"last_task3_{key}": value
                    for key, value in averages.items()
                },
            }
            if validation is not None:
                summary_payload["last_task3_validation"] = validation[
                    "loss_total"
                ]
                summary_payload["best_task3_validation"] = best_validation
            update_summary(summary_payload)

        if is_main_process() and current_epoch % int(train_cfg.get("save_every", 1)) == 0:
            checkpoint_path = weights_dir / f"checkpoint_epoch{current_epoch}.pth"
            checkpoint_payload = {
                "epoch": current_epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer_G": model.optimizer_G.state_dict(),
                "optimizer_D": model.optimizer_D.state_dict(),
                "optimizer_F": model.optimizer_F.state_dict(),
                "scheduler_G": scheduler_G.state_dict(),
                "scheduler_D": scheduler_D.state_dict(),
                "scheduler_F": scheduler_F.state_dict(),
                "best_validation": best_validation,
                "validation": validation,
                "data_split": {
                    "training_subjects": training_subjects,
                    "validation_subjects": validation_subjects,
                },
            }
            torch.save(checkpoint_payload, checkpoint_path)
            print(f"Saved: {checkpoint_path}")
            _log_checkpoint(
                stage="retro_scratch",
                path=checkpoint_path,
                epoch=current_epoch,
                global_step=global_step,
                validation=validation,
                is_best=is_best if validation is not None else None,
            )
            update_summary({"latest_checkpoint": str(checkpoint_path)})

            if validation is not None and is_best:
                best_path = weights_dir / "checkpoint_best_validation.pth"
                torch.save(checkpoint_payload, best_path)
                _audit(
                    f"[BestCheckpoint] Stage:retro_scratch, "
                    f"Epoch:{current_epoch}, GlobalStep:{global_step}, "
                    f"ValLoss:{validation['loss_total']:.8f}, "
                    f"SavedTo:{best_path.resolve()}"
                )
                update_summary({
                    "best_task3_checkpoint": str(best_path),
                    "best_task3_validation": best_validation,
                })

def _discover_cached_subjects(
    preprocessed_dir: str | Path,
    split: str,
    modalities: Sequence[str],
    fields: Sequence[str],
) -> List[str]:
    root = Path(preprocessed_dir)
    subjects = set()
    for modality in modalities:
        for field in fields:
            for path in (root / split / modality / field).glob("*.npz"):
                subjects.add(cached_subject_id(path, split, modality, field))
    return sorted(subjects)


def _paired_task3_dataset(
    cfg: dict,
    include_subjects: Optional[Iterable[str]] = None,
):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    modalities = list(data_cfg.get("modalities", ["T1W", "T2W", "T2FLAIR"]))
    fields = list(data_cfg.get("field_strengths", ["0.1T", "1.5T", "3T", "5T", "7T"]))
    crop_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_raw) if crop_raw else None
    split = train_cfg.get("split", "pro_train")
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    if not preprocessed_dir:
        raise RuntimeError(
            "The audited fine-tuning pipeline requires cached NPZ data."
        )

    datasets: List = []
    for modality in modalities:
        for source_field in fields:
            for target_field in fields:
                if source_field == target_field:
                    continue
                try:
                    base = CachedPairedDataset(
                        preprocessed_dir=preprocessed_dir,
                        split=split,
                        modality=modality,
                        source_field=source_field,
                        target_field=target_field,
                        crop_size=crop_size,
                        include_subjects=include_subjects,
                    )
                except (FileNotFoundError, ValueError):
                    continue
                datasets.append(
                    ConditionedPairedDataset(
                        base, modality, source_field, target_field
                    )
                )

    if not datasets:
        raise FileNotFoundError(
            "No prospective Task 3 paired field combinations were found."
        )
    return ConcatDataset(datasets)


def _paired_loss_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    criterion_l1,
    criterion_lpips,
    criterion_ssim,
    criterion_edge,
    lambda_l1: float,
    lambda_lpips: float,
    lambda_ssim: float,
    lambda_edge: float,
) -> Tuple[torch.Tensor, dict]:
    loss_l1 = criterion_l1(prediction, target)
    loss_lpips = criterion_lpips(prediction, target)
    loss_ssim = criterion_ssim(prediction, target)
    loss_edge = (
        criterion_edge(prediction, target)
        if criterion_edge is not None
        else torch.zeros((), device=prediction.device)
    )
    total = (
        lambda_l1 * loss_l1
        + lambda_lpips * loss_lpips
        + lambda_ssim * loss_ssim
        + lambda_edge * loss_edge
    )
    mse = torch.mean((prediction - target) ** 2)
    rmse = torch.sqrt(torch.clamp(mse, min=1e-12))
    psnr = 20.0 * torch.log10(
        torch.tensor(2.0, device=prediction.device) / rmse
    )
    values = {
        "loss_total": float(total.detach()),
        "loss_l1": float(loss_l1.detach()),
        "loss_lpips": float(loss_lpips.detach()),
        "loss_ssim": float(loss_ssim.detach()),
        "loss_edge": float(loss_edge.detach()),
        "mae": float(loss_l1.detach()),
        "rmse": float(rmse.detach()),
        "psnr": float(psnr.detach()),
        "ssim": float((1.0 - loss_ssim).detach()),
    }
    return total, values


def _validate_task3_finetune(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    *,
    criterion_l1,
    criterion_lpips,
    criterion_ssim,
    criterion_edge,
    lambda_l1: float,
    lambda_lpips: float,
    lambda_ssim: float,
    lambda_edge: float,
) -> dict:
    with _preserve_training_rng_state():
        was_training = generator.training
        generator.eval()
        sums = {
            "loss_total": 0.0,
            "loss_l1": 0.0,
            "loss_lpips": 0.0,
            "loss_ssim": 0.0,
            "loss_edge": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
        }
        completed = 0
        with torch.no_grad():
            for iteration, batch in enumerate(loader):
                if max_batches and iteration >= max_batches:
                    break
                source = batch["source"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                source_ids, target_ids, modality_ids = _conditions(batch, device)
                prediction = generator(
                    source,
                    source_field_id=source_ids,
                    target_field_id=target_ids,
                    modality_id=modality_ids,
                )
                _, values = _paired_loss_values(
                    prediction,
                    target,
                    criterion_l1=criterion_l1,
                    criterion_lpips=criterion_lpips,
                    criterion_ssim=criterion_ssim,
                    criterion_edge=criterion_edge,
                    lambda_l1=lambda_l1,
                    lambda_lpips=lambda_lpips,
                    lambda_ssim=lambda_ssim,
                    lambda_edge=lambda_edge,
                )
                for key in sums:
                    sums[key] += values[key]
                completed += 1
        if was_training:
            generator.train()
        return {key: value / max(completed, 1) for key, value in sums.items()}

def train_task3_conditioned_finetune(
    cfg: dict,
    device: torch.device,
    use_dist: bool = False,
    resume_ckpt: str | None = None,
) -> None:
    """Audited paired fine-tuning over every ordered field pair."""
    if use_dist:
        raise NotImplementedError(
            "The final audited pipeline is intentionally single-GPU."
        )
    if os.environ.get("MRI_XFIELDS_PROSPECTIVE_DATA_CONFIRMED") != "YES":
        raise RuntimeError(
            "Prospective data use has not been confirmed. Set "
            "MRI_XFIELDS_PROSPECTIVE_DATA_CONFIRMED=YES only after verifying "
            "that the directory is not the withdrawn 20-volunteer dataset."
        )

    model_cfg = _normalize_task3_model_config(cfg["model"])
    cfg["model"] = model_cfg
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    pretrain_cfg = cfg.get("pretrain", {})
    finetune_cfg = cfg.get("finetune", {})
    validation_cfg = _validation_config(cfg)
    augmentation_cfg = _finetune_augmentation_config(finetune_cfg)

    output_dir = Path(train_cfg["output_dir"])
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    generator = build_generator(model_cfg, data_cfg=data_cfg, task=3).to(device)
    pretrained = pretrain_cfg.get("checkpoint")
    use_conditional_discriminator = bool(
        finetune_cfg.get("conditional_discriminator", False)
    )
    lambda_gan = float(finetune_cfg.get("loss_gan", 0.01))
    discriminator_lr = float(
        finetune_cfg.get("discriminator_lr", finetune_cfg.get("lr", 5e-5))
    )
    finetune_gradient_clip_norm = float(
        finetune_cfg.get("gradient_clip_norm", 0.0) or 0.0
    )

    state = None
    if pretrained:
        state = audited_torch_load(
            pretrained, torch, map_location=device, weights_only=False
        )
        generator.load_state_dict(_extract_generator_state(state))
        if is_main_process():
            print(f"Loaded Task 3 UNETR pretraining from: {pretrained}")
    else:
        initialize_generator(generator)
        if is_main_process():
            print("Task 3 UNETR fine-tuning from random initialization")

    discriminator = None
    criterion_gan = None
    optimizer_D = None
    if use_conditional_discriminator:
        if state is None:
            raise RuntimeError(
                "Stage-2 conditional GAN requires a Stage-1 checkpoint containing netD."
            )
        discriminator = ConditionalPatchDiscriminator(
            image_channels=model_cfg.get("output_nc", 1),
            ndf=model_cfg.get("ndf", 64),
        ).to(device)
        discriminator.load_state_dict(_extract_discriminator_state(state))
        criterion_gan = GANLoss(mode=model_cfg.get("gan_mode", "lsgan")).to(device)
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=discriminator_lr,
            betas=(
                finetune_cfg.get("beta1", 0.5),
                finetune_cfg.get("beta2", 0.999),
            ),
        )
        if is_main_process():
            print(
                "Stage 2 conditional discriminator ENABLED: "
                f"lambda_GAN={lambda_gan}, lr_D={discriminator_lr:.3e}"
            )

    modalities = list(data_cfg.get("modalities", ["T1W", "T2W", "T2FLAIR"]))
    fields = list(data_cfg.get("field_strengths", ["0.1T", "1.5T", "3T", "5T", "7T"]))
    split = train_cfg.get("split", "pro_train")
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    subjects = _discover_cached_subjects(
        preprocessed_dir, split, modalities, fields
    )
    all_subjects = sorted(subjects)
    training_subjects, validation_subjects = _split_subjects(
        all_subjects, validation_cfg["fraction"], validation_cfg["seed"]
    )

    # Stage 2 uses a true subject-disjoint holdout because validation controls
    # LR scheduling and early stopping.
    training_dataset = _paired_task3_dataset(
        cfg, include_subjects=training_subjects
    )
    validation_dataset = _paired_task3_dataset(
        cfg, include_subjects=validation_subjects
    )
    _write_data_split(
        output_dir,
        stage="pro_pretrained",
        source_split=split,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        seed=validation_cfg["seed"],
        fraction=validation_cfg["fraction"],
        split_method="deterministic subject-level holdout",
        validation_overlaps_training=False,
    )

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_options = _audited_loader_kwargs(
        num_workers,
        pin_memory=device.type == "cuda",
    )
    loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_cfg["batch_size"],
        shuffle=False,
        **loader_options,
    )
    criterion_l1 = nn.L1Loss()
    criterion_lpips = PerceptualLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)
    lambda_l1 = float(finetune_cfg.get("loss_l1", 1.0))
    lambda_lpips = float(finetune_cfg.get("loss_lpips", 0.1))
    lambda_ssim = float(finetune_cfg.get("loss_ssim", 0.0))
    lambda_edge = float(finetune_cfg.get("loss_edge", 0.0))
    criterion_edge = None
    if lambda_edge > 0:
        criterion_edge = StructureLoss(
            ssim_weight=0.0, edge_weight=1.0
        ).to(device)

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg.get("lr", 5e-5),
        betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
    )

    # Validation affects future Stage 2 training through scheduler / stopping,
    # but validation samples never contribute gradients.
    val_lr_factor = float(finetune_cfg.get("val_lr_factor", 0.5))
    val_lr_patience = int(finetune_cfg.get("val_lr_patience", 5))
    val_min_lr = float(finetune_cfg.get("val_min_lr", 1e-7))
    val_min_delta = float(finetune_cfg.get("val_min_delta", 1e-4))
    early_stopping_patience = int(
        finetune_cfg.get("early_stopping_patience", 10)
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=val_lr_factor,
        patience=val_lr_patience,
        threshold=val_min_delta,
        threshold_mode="abs",
        min_lr=val_min_lr,
    )

    start_epoch = 0
    best_validation = float("inf")
    epochs_without_improvement = 0
    if resume_ckpt:
        state = audited_torch_load(resume_ckpt, torch, map_location=device, weights_only=False)
        generator.load_state_dict(state.get("generator", state))
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if use_conditional_discriminator:
            if "discriminator" not in state:
                raise RuntimeError(
                    "Cannot resume conditional Stage-2 GAN: checkpoint has no discriminator."
                )
            discriminator.load_state_dict(state["discriminator"])
            if "optimizer_D" in state and optimizer_D is not None:
                optimizer_D.load_state_dict(state["optimizer_D"])
        if "plateau_scheduler" in state:
            plateau_scheduler.load_state_dict(state["plateau_scheduler"])
        start_epoch = int(state.get("epoch", 0))
        best_validation = float(
            state.get("best_validation", float("inf"))
        )
        epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )

    epochs = int(train_cfg.get("epochs", 100))
    max_iters = _runtime_max_iters(train_cfg, len(loader))
    steps_per_epoch = min(len(loader), max_iters) if max_iters else len(loader)
    global_step = start_epoch * max(steps_per_epoch, 1)

    _audit(
        f"[StageStart] Stage:pro_pretrained, InitializationCheckpoint:{pretrained}, "
        f"EpochStart:{start_epoch}, TotalEpochs:{epochs}, "
        f"BatchSize:{batch_size}, StepsPerEpoch:{steps_per_epoch}, "
        f"Augmentation:{json.dumps(augmentation_cfg, sort_keys=True)}, "
        f"ConditionalDiscriminator:{use_conditional_discriminator}, "
        f"LambdaGAN:{lambda_gan if use_conditional_discriminator else 0.0}, "
        f"DiscriminatorLR:{discriminator_lr if use_conditional_discriminator else 0.0}, "
        f"TrainSubjects:{len(training_subjects)}, "
        f"ValidationSubjects:{len(validation_subjects)}, "
        f"RuntimeDataFraction:{os.environ.get('MRI_XFIELDS_DATA_FRACTION', '1.0')}, "
        "ProspectiveDataConfirmation:YES, "
        "ValidationExecution:every_epoch_controls_lr_and_early_stopping, "
        f"ValLRFactor:{val_lr_factor}, ValLRPatience:{val_lr_patience}, "
        f"EarlyStoppingPatience:{early_stopping_patience}"
    )

    # Establish the validation baseline of the pretrained generator.
    if start_epoch == 0 and not resume_ckpt:
        initial_validation = _validate_task3_finetune(
            generator,
            validation_loader,
            device,
            validation_cfg["max_batches"],
            criterion_l1=criterion_l1,
            criterion_lpips=criterion_lpips,
            criterion_ssim=criterion_ssim,
            criterion_edge=criterion_edge,
            lambda_l1=lambda_l1,
            lambda_lpips=lambda_lpips,
            lambda_ssim=lambda_ssim,
            lambda_edge=lambda_edge,
        )
        best_validation = initial_validation["loss_total"]
        _log_validation(
            stage="pro_pretrained",
            epoch=0,
            global_step=0,
            values=initial_validation,
            event="initial_pretrained_performance",
        )

        if is_main_process():
            initial_checkpoint_path = weights_dir / "checkpoint_epoch0.pth"
            initial_payload = {
                "epoch": 0,
                "global_step": 0,
                "generator": generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "plateau_scheduler": plateau_scheduler.state_dict(),
                "best_validation": best_validation,
                "epochs_without_improvement": 0,
                "validation": initial_validation,
                "pretrain_checkpoint": pretrained,
                "stage2_conditional_discriminator": use_conditional_discriminator,
                "lambda_gan": lambda_gan if use_conditional_discriminator else 0.0,
                "data_split": {
                    "training_subjects": training_subjects,
                    "validation_subjects": validation_subjects,
                },
            }
            if use_conditional_discriminator:
                initial_payload["discriminator"] = discriminator.state_dict()
                initial_payload["optimizer_D"] = optimizer_D.state_dict()
            torch.save(initial_payload, initial_checkpoint_path)
            best_path = weights_dir / "checkpoint_best_validation.pth"
            torch.save(initial_payload, best_path)
            print(
                f"Initial Stage 2 validation: "
                f"{initial_validation['loss_total']:.6f}"
            )
            _log_checkpoint(
                stage="pro_pretrained",
                path=initial_checkpoint_path,
                epoch=0,
                global_step=0,
                validation=initial_validation,
                is_best=True,
            )
            update_summary({
                "best_task3_finetune_validation": best_validation,
                "best_task3_finetune_checkpoint": str(best_path),
            })

    for epoch in range(start_epoch, epochs):
        generator.train()
        if discriminator is not None:
            discriminator.train()
        total = 0.0
        total_d = 0.0
        completed = 0
        progress = tqdm(
            loader,
            desc=f"Task3 finetune {epoch + 1}/{epochs}",
            disable=not is_main_process(),
        )
        for iteration, batch in enumerate(progress):
            if max_iters and iteration >= max_iters:
                break
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            source, target = _augment_paired_finetune_batch(
                source, target, augmentation_cfg
            )
            source_ids, target_ids, modality_ids = _conditions(batch, device)
            prediction = generator(
                source,
                source_field_id=source_ids,
                target_field_id=target_ids,
                modality_id=modality_ids,
            )
            reconstruction_loss, values = _paired_loss_values(
                prediction,
                target,
                criterion_l1=criterion_l1,
                criterion_lpips=criterion_lpips,
                criterion_ssim=criterion_ssim,
                criterion_edge=criterion_edge,
                lambda_l1=lambda_l1,
                lambda_lpips=lambda_lpips,
                lambda_ssim=lambda_ssim,
                lambda_edge=lambda_edge,
            )

            loss_d_value = 0.0
            loss_g_gan_value = 0.0
            loss = reconstruction_loss

            if use_conditional_discriminator:
                # 1) Update the pretrained conditional PatchGAN.
                for parameter in discriminator.parameters():
                    parameter.requires_grad = True
                optimizer_D.zero_grad(set_to_none=True)
                prediction_fake_d = discriminator(
                    prediction.detach(), target_ids, modality_ids
                )
                prediction_real_d = discriminator(
                    target, target_ids, modality_ids
                )
                loss_d_fake = criterion_gan(prediction_fake_d, False).mean()
                loss_d_real = criterion_gan(prediction_real_d, True).mean()
                loss_d = 0.5 * (loss_d_fake + loss_d_real)
                loss_d.backward()
                if finetune_gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), finetune_gradient_clip_norm
                    )
                optimizer_D.step()
                loss_d_value = float(loss_d.detach())

                # 2) Update G with paired reconstruction/perceptual loss plus
                #    a low-weight conditional adversarial term.
                for parameter in discriminator.parameters():
                    parameter.requires_grad = False
                prediction_fake_g = discriminator(
                    prediction, target_ids, modality_ids
                )
                loss_g_gan = criterion_gan(prediction_fake_g, True).mean()
                loss = reconstruction_loss + lambda_gan * loss_g_gan
                loss_g_gan_value = float(loss_g_gan.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if finetune_gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    generator.parameters(), finetune_gradient_clip_norm
                )
            optimizer.step()

            values["loss_reconstruction_total"] = values["loss_total"]
            values["loss_G_GAN"] = loss_g_gan_value
            values["loss_D"] = loss_d_value
            values["loss_total"] = float(loss.detach())

            total += values["loss_total"]
            total_d += loss_d_value
            completed += 1
            global_step += 1
            if is_main_process():
                progress.set_postfix(loss=f"{values['loss_total']:.4f}")
                _audit(
                    f"[Training] Stage:pro_pretrained, Epoch:{epoch + 1}, "
                    f"Iteration:{iteration + 1}, GlobalStep:{global_step}, "
                    f"LR:{optimizer.param_groups[0]['lr']}, Scheduler:ReduceLROnPlateau, "
                    f"BatchSize:{source.shape[0]}, "
                    f"LossTotal:{values['loss_total']:.8f}, "
                    f"Losses:{json.dumps(values, sort_keys=True)}, "
                    f"LossWeights:{json.dumps({'L1': lambda_l1, 'LPIPS': lambda_lpips, 'SSIM': lambda_ssim, 'Edge': lambda_edge, 'GAN': lambda_gan if use_conditional_discriminator else 0.0})}, "
                    f"Augmentation:{json.dumps(augmentation_cfg, sort_keys=True)}, "
                    f"MemUsage:{get_mem_usage()}"
                )
                if should_log(global_step):
                    payload = {
                        f"task3_finetune/{key}": value
                        for key, value in values.items()
                    }
                    payload.update({
                        "task3_finetune/lr": optimizer.param_groups[0]["lr"],
                        "task3_finetune/epoch": epoch + 1,
                    })
                    if should_log_images(global_step):
                        payload.update(
                            _prediction_diagnostics(
                                "task3_finetune_prediction_stats", prediction
                            )
                        )
                        payload.update(
                            image_payload(
                                source=source,
                                prediction=prediction,
                                target=target,
                                prefix="task3_finetune_images",
                            )
                        )
                    log_wandb(payload, step=global_step)

        average = total / max(completed, 1)
        average_d = total_d / max(completed, 1)
        current_epoch = epoch + 1

        validation = _validate_task3_finetune(
            generator,
            validation_loader,
            device,
            validation_cfg["max_batches"],
            criterion_l1=criterion_l1,
            criterion_lpips=criterion_lpips,
            criterion_ssim=criterion_ssim,
            criterion_edge=criterion_edge,
            lambda_l1=lambda_l1,
            lambda_lpips=lambda_lpips,
            lambda_ssim=lambda_ssim,
            lambda_edge=lambda_edge,
        )
        _log_validation(
            stage="pro_pretrained",
            epoch=current_epoch,
            global_step=global_step,
            values=validation,
            event="epoch_end_training_control",
        )

        previous_lr = optimizer.param_groups[0]["lr"]
        plateau_scheduler.step(validation["loss_total"])
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = (
            validation["loss_total"] < best_validation - val_min_delta
        )
        if is_best:
            best_validation = validation["loss_total"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        _audit(
            f"[ValidationControl] Stage:pro_pretrained, "
            f"Epoch:{current_epoch}, GlobalStep:{global_step}, "
            f"ValLoss:{validation['loss_total']:.8f}, "
            f"BestValLoss:{best_validation:.8f}, IsBest:{is_best}, "
            f"EpochsWithoutImprovement:{epochs_without_improvement}, "
            f"LRBefore:{previous_lr}, LRAfter:{current_lr}, "
            f"EarlyStoppingPatience:{early_stopping_patience}"
        )

        if is_main_process():
            print(
                f"Task3 finetune epoch {current_epoch}: "
                f"train={average:.6f} "
                f"D={average_d:.6f} "
                f"val={validation['loss_total']:.6f} "
                f"lr={current_lr:.3e} "
                f"no_improve={epochs_without_improvement}/"
                f"{early_stopping_patience}"
            )
            log_wandb(
                {
                    "task3_finetune_epoch/avg_loss": average,
                    "task3_finetune_epoch/avg_loss_D": average_d,
                    "task3_finetune_epoch/index": current_epoch,
                    "task3_finetune_epoch/lr": current_lr,
                    "task3_finetune_epoch/best_val_loss": best_validation,
                    "task3_finetune_epoch/epochs_without_improvement":
                        epochs_without_improvement,
                    **{
                        f"task3_finetune_validation/{key}": value
                        for key, value in validation.items()
                    },
                },
                step=global_step,
            )
            update_summary({
                "last_task3_finetune_epoch": current_epoch,
                "last_task3_finetune_loss": average,
                "last_task3_finetune_validation":
                    validation["loss_total"],
                "best_task3_finetune_validation": best_validation,
            })

        checkpoint_payload = {
            "epoch": current_epoch,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "plateau_scheduler": plateau_scheduler.state_dict(),
            "best_validation": best_validation,
            "epochs_without_improvement": epochs_without_improvement,
            "validation": validation,
            "pretrain_checkpoint": pretrained,
            "stage2_conditional_discriminator": use_conditional_discriminator,
            "lambda_gan": lambda_gan if use_conditional_discriminator else 0.0,
            "data_split": {
                "training_subjects": training_subjects,
                "validation_subjects": validation_subjects,
            },
        }
        if use_conditional_discriminator:
            checkpoint_payload["discriminator"] = discriminator.state_dict()
            checkpoint_payload["optimizer_D"] = optimizer_D.state_dict()

        if is_main_process() and current_epoch % int(train_cfg.get("save_every", 1)) == 0:
            checkpoint_path = weights_dir / f"checkpoint_epoch{current_epoch}.pth"
            torch.save(checkpoint_payload, checkpoint_path)
            print(f"Saved: {checkpoint_path}")
            _log_checkpoint(
                stage="pro_pretrained",
                path=checkpoint_path,
                epoch=current_epoch,
                global_step=global_step,
                validation=validation,
                is_best=is_best,
            )
            update_summary({"latest_checkpoint": str(checkpoint_path)})

            if is_best:
                best_path = weights_dir / "checkpoint_best_validation.pth"
                torch.save(checkpoint_payload, best_path)
                _audit(
                    f"[BestCheckpoint] Stage:pro_pretrained, "
                    f"Epoch:{current_epoch}, GlobalStep:{global_step}, "
                    f"ValLoss:{validation['loss_total']:.8f}, "
                    f"SavedTo:{best_path.resolve()}"
                )
                update_summary({
                    "best_task3_finetune_checkpoint": str(best_path)
                })

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            _audit(
                f"[EarlyStopping] Stage:pro_pretrained, "
                f"Epoch:{current_epoch}, GlobalStep:{global_step}, "
                f"BestValLoss:{best_validation:.8f}, "
                f"Patience:{early_stopping_patience}"
            )
            if is_main_process():
                print(
                    f"Early stopping Stage 2 at epoch {current_epoch}: "
                    f"validation did not improve for "
                    f"{epochs_without_improvement} epochs."
                )
            break

