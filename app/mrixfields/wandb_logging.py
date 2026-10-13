"""Optional Weights & Biases logging for MRIxFields training.

The module imports ``wandb`` lazily, so normal training remains unchanged when
tracking is disabled. Only the main DDP process initializes and writes a run.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_RUN = None
_WANDB = None
_LOG_EVERY = 50
_IMAGE_EVERY = 500


def init_wandb(
    cfg: Dict[str, Any],
    output_dir: str | Path,
    *,
    enabled: bool,
    project: str,
    entity: Optional[str],
    run_name: str,
    mode: str = "online",
    log_every: int = 50,
    image_every: int = 500,
):
    """Initialize one resumable W&B run and return it.

    A stable run ID is saved next to the checkpoints. Resubmitting the same
    Snellius experiment therefore resumes the same W&B run as well as the same
    training checkpoint.
    """
    global _RUN, _WANDB, _LOG_EVERY, _IMAGE_EVERY

    if not enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking was requested but the 'wandb' package is not "
            "installed. Install it with: python -m pip install wandb"
        ) from exc

    output_dir = Path(output_dir)
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run_id_path = output_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id + "\n", encoding="utf-8")

    _LOG_EVERY = max(1, int(log_every))
    _IMAGE_EVERY = max(0, int(image_every))
    _WANDB = wandb
    _RUN = wandb.init(
        project=project,
        entity=entity or None,
        name=run_name,
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(wandb_dir),
        config=cfg,
        save_code=True,
        tags=[
            str(cfg.get("method", "unknown")),
            str(cfg.get("task_name", "unknown")),
            str(cfg.get("_runtime", {}).get("mode", "unknown")),
        ],
    )

    atexit.register(finish_wandb)
    print(f"W&B run: {_RUN.url}", flush=True)
    return _RUN


def wandb_enabled() -> bool:
    return _RUN is not None


def should_log(step: int) -> bool:
    return _RUN is not None and step % _LOG_EVERY == 0


def should_log_images(step: int) -> bool:
    return (
        _RUN is not None
        and _IMAGE_EVERY > 0
        and step % _IMAGE_EVERY == 0
    )


def log_wandb(payload: Dict[str, Any], *, step: int) -> None:
    if _RUN is not None:
        _RUN.log(payload, step=step)


def _first_mri_slice(tensor: torch.Tensor):
    image = tensor.detach().float().cpu()
    while image.ndim > 2:
        image = image[0]
    minimum = image.min()
    maximum = image.max()
    if float(maximum - minimum) > 1e-8:
        image = (image - minimum) / (maximum - minimum)
    else:
        image = torch.zeros_like(image)
    return image.numpy()


def image_payload(
    *,
    source: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    prefix: str,
) -> Dict[str, Any]:
    if _RUN is None or _WANDB is None:
        return {}
    return {
        f"{prefix}/source": _WANDB.Image(_first_mri_slice(source)),
        f"{prefix}/prediction": _WANDB.Image(_first_mri_slice(prediction)),
        f"{prefix}/target": _WANDB.Image(_first_mri_slice(target)),
    }


def update_summary(values: Dict[str, Any]) -> None:
    if _RUN is not None:
        for key, value in values.items():
            _RUN.summary[key] = value


def finish_wandb() -> None:
    global _RUN
    if _RUN is not None:
        try:
            _RUN.finish()
        finally:
            _RUN = None
