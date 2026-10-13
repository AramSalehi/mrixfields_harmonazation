#!/usr/bin/env python3
"""MRIxFields2026 manifest-driven Docker inference entrypoint.

Reads /input/manifest.json and writes exactly the predictions requested by
manifest.samples[*].output under /output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from mrixfields.models.generator_factory import build_generator, forward_generator
from mrixfields.data.transforms import CenterCropOrPad


INPUT_ROOT = Path(os.environ.get("MRIX_INPUT_ROOT", "/input"))
OUTPUT_ROOT = Path(os.environ.get("MRIX_OUTPUT_ROOT", "/output"))
MANIFEST_PATH = INPUT_ROOT / "manifest.json"
CONFIG_PATH = Path(os.environ.get("MRIX_CONFIG", "/app/config.yaml"))
CHECKPOINT_PATH = Path(os.environ.get("MRIX_CHECKPOINT", "/app/weights/model.pth"))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_generator_state(checkpoint: object) -> dict:
    """Extract generator weights from training/fine-tuning checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)!r}")

    if "model" in checkpoint:
        model_state = checkpoint["model"]
        for prefix in ("netG.module.", "netG.", "netG_AB.module.", "netG_AB."):
            gen_state = {
                k[len(prefix):]: v
                for k, v in model_state.items()
                if k.startswith(prefix)
            }
            if gen_state:
                return gen_state
        # Some checkpoints store the generator directly under model.
        if model_state and all(isinstance(k, str) for k in model_state):
            return model_state

    if "generator" in checkpoint:
        return checkpoint["generator"]

    # Raw state dict.
    if checkpoint and all(isinstance(k, str) for k in checkpoint):
        tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
        if tensor_values:
            return checkpoint

    raise ValueError(f"Unrecognized checkpoint format. Top-level keys: {list(checkpoint.keys())}")


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_generator(
        cfg["model"],
        data_cfg=cfg.get("data", {}),
        task=int(cfg.get("task", 3)),
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = extract_generator_state(checkpoint)

    # Handle a fully DDP-prefixed raw generator state dict if present.
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model


def predict_canonical_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    source_field: str,
    target_field: str,
    modality: str,
    task_number: int,
    crop_size: tuple[int, int],
    slice_axis: int,
    device: torch.device,
) -> np.ndarray:
    """Run full-volume 2D slice inference in canonical orientation."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    crop = CenterCropOrPad(crop_size)
    output = np.zeros(volume.shape, dtype=np.float32)
    original_shape = volume.shape

    if slice_axis == 2:
        original_slice_shape = (original_shape[0], original_shape[1])
    elif slice_axis == 1:
        original_slice_shape = (original_shape[0], original_shape[2])
    elif slice_axis == 0:
        original_slice_shape = (original_shape[1], original_shape[2])
    else:
        raise ValueError(f"Invalid slice_axis={slice_axis}")
    uncrop = CenterCropOrPad(original_slice_shape)

    n_slices = volume.shape[slice_axis]
    with torch.inference_mode():
        for i in range(n_slices):
            slicing = [slice(None)] * 3
            slicing[slice_axis] = i
            s = tuple(slicing)

            slc = np.asarray(volume[s], dtype=np.float32)
            slc_input = crop(slc)
            slc_scaled = slc_input * 2.0 - 1.0
            tensor = torch.from_numpy(slc_scaled).unsqueeze(0).unsqueeze(0).to(device)

            # IMPORTANT: explicitly condition every Task-3 prediction from the manifest.
            pred = forward_generator(
                model,
                tensor,
                source_field=source_field,
                target_field=target_field,
                modality=modality,
                task=task_number,
            )
            pred_np = pred.squeeze(0).squeeze(0).float().cpu().numpy()
            pred_np = uncrop(pred_np)
            pred_np = np.clip(pred_np, -1.0, 1.0) * 0.5 + 0.5
            output[s] = pred_np.astype(np.float32, copy=False)

    return output


def process_sample(model, cfg: dict, sample: dict, device: torch.device) -> None:
    required = ("input", "output", "source_field", "target_field", "modality")
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError(f"Manifest sample is missing keys: {missing}")

    input_path = INPUT_ROOT / sample["input"]
    output_path = OUTPUT_ROOT / sample["output"]
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    original_img = nib.load(str(input_path))
    original_data = original_img.get_fdata(dtype=np.float32)
    original_ornt = nib.io_orientation(original_img.affine)

    canonical_img = nib.as_closest_canonical(original_img)
    canonical_data = canonical_img.get_fdata(dtype=np.float32)
    canonical_ornt = nib.io_orientation(canonical_img.affine)

    if not np.isfinite(canonical_data).all():
        raise ValueError(f"Input contains NaN/Inf: {input_path}")

    crop_size = tuple(int(x) for x in cfg.get("data", {}).get("crop_size", [384, 448]))
    slice_axis = int(cfg.get("data", {}).get("slice_axis", 2))
    task_token = str(sample.get("task", "task3"))
    task_number = int(task_token.replace("task", "")) if task_token.startswith("task") else int(cfg.get("task", 3))

    pred = predict_canonical_volume(
        model=model,
        volume=canonical_data,
        source_field=str(sample["source_field"]),
        target_field=str(sample["target_field"]),
        modality=str(sample["modality"]),
        task_number=task_number,
        crop_size=crop_size,
        slice_axis=slice_axis,
        device=device,
    )

    # Return prediction from canonical orientation to the exact input orientation.
    transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
    pred = nib.orientations.apply_orientation(pred, transform)

    # Match the project's existing inference behavior for skull-stripped inputs.
    pred *= (original_data > 1e-6).astype(np.float32)
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)

    if pred.shape != original_data.shape:
        raise RuntimeError(
            f"Output shape {pred.shape} does not match input shape {original_data.shape} for {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(pred, original_img.affine, header=original_img.header.copy())
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, str(output_path))
    print(f"WROTE {output_path} shape={pred.shape} range=[{pred.min():.6f}, {pred.max():.6f}]", flush=True)


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Required manifest not found: {MANIFEST_PATH}")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}. "
            "Copy your trained Task-3 checkpoint to app/weights/model.pth before docker build."
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest.json must contain a non-empty 'samples' list")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={device}", flush=True)
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable; inference will run on CPU.", flush=True)

    cfg = load_yaml(CONFIG_PATH)
    model = load_model(cfg, CHECKPOINT_PATH, device)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}", flush=True)
    print(f"Manifest task={manifest.get('task')} samples={len(samples)}", flush=True)

    expected_outputs = set()
    for idx, sample in enumerate(samples, start=1):
        rel_out = str(sample.get("output", ""))
        if not rel_out or rel_out in expected_outputs:
            raise ValueError(f"Missing or duplicate output path in manifest at sample {idx}: {rel_out!r}")
        expected_outputs.add(rel_out)
        print(
            f"[{idx}/{len(samples)}] {sample.get('modality')} "
            f"{sample.get('source_field')}->{sample.get('target_field')} "
            f"case={sample.get('case_id')}",
            flush=True,
        )
        process_sample(model, cfg, sample, device)

    # Final exact-output validation: every manifest output exists and no extra NIfTI predictions exist.
    actual_outputs = {
        str(p.relative_to(OUTPUT_ROOT))
        for p in OUTPUT_ROOT.rglob("*.nii.gz")
    }
    missing = sorted(expected_outputs - actual_outputs)
    extra = sorted(actual_outputs - expected_outputs)
    if missing or extra:
        raise RuntimeError(f"Output-set mismatch. missing={missing}, extra={extra}")

    print(f"SUCCESS: wrote exactly {len(expected_outputs)} predictions.", flush=True)


if __name__ == "__main__":
    main()
