"""Training loops for the Task 3 conditioned-UNETR CUT method."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data.cached_dataset import CachedPairedDataset
from .data.dataset import PairedMRIDataset
from .data.task3_dataset import (
    CachedTask3UnpairedDataset,
    ConditionedPairedDataset,
    Task3UnpairedMRIDataset,
)
from .data.utils import ABBR_TO_SPLIT as SPLIT_MAP
from .losses.perceptual import PerceptualLoss
from .losses.structure import SSIMLoss, StructureLoss
from .models.conditioned_unetr import FIELD_TO_ID, MODALITY_TO_ID
from .models.generator_factory import build_generator, initialize_generator
from .models.task3_cut_model import Task3ConditionedCUTModel, condition_ids
from .utils_dist import get_dist_info, is_main_process
from .wandb_logging import (
    image_payload,
    log_wandb,
    should_log,
    should_log_images,
    update_summary,
)


def _lr_lambda(n_epochs: int, n_epochs_decay: int):
    def rule(epoch: int) -> float:
        return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
    return rule


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


def train_task3_conditioned_cut(
    cfg: dict,
    device: torch.device,
    use_dist: bool = False,
    resume_ckpt: str | None = None,
) -> None:
    """Unpaired retrospective pretraining for one universal Task 3 UNETR."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    _, world_size = get_dist_info()

    output_dir = Path(train_cfg["output_dir"])
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    model = Task3ConditionedCUTModel(
        model_config=model_cfg,
        data_config=data_cfg,
        lr=train_cfg.get("lr", 2e-4),
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

    if preprocessed_dir:
        dataset = CachedTask3UnpairedDataset(
            preprocessed_dir=preprocessed_dir,
            split=split,
            modalities=modalities,
            field_strengths=fields,
            crop_size=crop_size,
        )
    else:
        dataset = Task3UnpairedMRIDataset(
            data_root=data_cfg["data_dir"],
            split=SPLIT_MAP.get(split, split),
            modalities=modalities,
            field_strengths=fields,
            crop_size=crop_size,
            slice_axis=data_cfg.get("slice_axis", 2),
        )

    batch_size = int(train_cfg.get("batch_size", 1))
    if use_dist:
        batch_size = max(1, batch_size // world_size)
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=train_cfg.get("num_workers", 4) > 0,
    )

    init_batch = next(iter(loader))
    init_source = init_batch["source"].to(device)
    source_ids, target_ids, modality_ids = _conditions(init_batch, device)
    model.data_dependent_initialize(
        init_source, source_ids, target_ids, modality_ids
    )
    model.setup_optimizers()

    if use_dist:
        model.netG = DDP(model.netG, device_ids=[device.index])
        model.netD = DDP(model.netD, device_ids=[device.index])
        model.netF = DDP(
            model.netF,
            device_ids=[device.index],
            find_unused_parameters=True,
        )

    n_epochs = int(train_cfg.get("n_epochs", 100))
    n_epochs_decay = int(train_cfg.get("n_epochs_decay", 0))
    rule = _lr_lambda(n_epochs, n_epochs_decay)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(model.optimizer_G, rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(model.optimizer_D, rule)
    scheduler_F = torch.optim.lr_scheduler.LambdaLR(model.optimizer_F, rule)

    start_epoch = 0
    if resume_ckpt:
        checkpoint = torch.load(
            resume_ckpt, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        if "optimizer_G" in checkpoint:
            model.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            model.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
            model.optimizer_F.load_state_dict(checkpoint["optimizer_F"])
            scheduler_G.load_state_dict(checkpoint["scheduler_G"])
            scheduler_D.load_state_dict(checkpoint["scheduler_D"])
            scheduler_F.load_state_dict(checkpoint["scheduler_F"])
        start_epoch = int(checkpoint.get("epoch", 0))
        if is_main_process():
            print(f"Resumed Task 3 conditioned CUT from {resume_ckpt} at epoch {start_epoch}")

    total_epochs = n_epochs + n_epochs_decay
    max_iters = int(train_cfg.get("max_iters_per_epoch", 0) or 0)
    steps_per_epoch = min(len(loader), max_iters) if max_iters else len(loader)
    global_step = start_epoch * max(steps_per_epoch, 1)

    if is_main_process():
        print(
            "Training Task 3 conditioned UNETR with unpaired CUT: "
            f"modalities={modalities}, fields={fields}, "
            f"epochs={start_epoch + 1}-{total_epochs}, "
            f"iters/epoch={len(loader)}"
            + (f" capped_at={max_iters}" if max_iters else "")
        )

    for epoch in range(start_epoch, total_epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
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
                source,
                target,
                source_ids,
                target_ids,
                modality_ids,
            )
            completed += 1
            global_step += 1
            for key in sums:
                sums[key] += losses[key]

            if is_main_process():
                progress.set_postfix(
                    G=f"{losses['loss_G']:.4f}",
                    D=f"{losses['loss_D']:.4f}",
                    NCE=f"{losses['loss_NCE']:.4f}",
                )
                if should_log(global_step):
                    payload = {
                        f"task3/{key}": value
                        for key, value in losses.items()
                    }
                    payload.update({
                        "task3/lr_G": model.optimizer_G.param_groups[0]["lr"],
                        "task3/lr_D": model.optimizer_D.param_groups[0]["lr"],
                        "task3/epoch": epoch + 1,
                    })
                    if should_log_images(global_step):
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

        if is_main_process():
            averages = {
                key: value / max(completed, 1)
                for key, value in sums.items()
            }
            print(
                f"Task3 epoch {epoch + 1}: "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in averages.items()
                )
            )
            log_wandb(
                {
                    **{f"task3_epoch/{key}": value for key, value in averages.items()},
                    "task3_epoch/index": epoch + 1,
                },
                step=global_step,
            )
            update_summary({
                "last_task3_epoch": epoch + 1,
                **{f"last_task3_{key}": value for key, value in averages.items()},
            })

        if is_main_process() and (epoch + 1) % int(train_cfg.get("save_every", 10)) == 0:
            path = weights_dir / f"checkpoint_epoch{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer_G": model.optimizer_G.state_dict(),
                    "optimizer_D": model.optimizer_D.state_dict(),
                    "optimizer_F": model.optimizer_F.state_dict(),
                    "scheduler_G": scheduler_G.state_dict(),
                    "scheduler_D": scheduler_D.state_dict(),
                    "scheduler_F": scheduler_F.state_dict(),
                },
                path,
            )
            print(f"Saved: {path}")
            update_summary({"latest_checkpoint": str(path)})


def _paired_task3_dataset(cfg: dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    modalities = list(data_cfg.get("modalities", ["T1W", "T2W", "T2FLAIR"]))
    fields = list(data_cfg.get("field_strengths", ["0.1T", "1.5T", "3T", "5T", "7T"]))
    crop_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_raw) if crop_raw else None
    split = train_cfg.get("split", "pro_train")
    preprocessed_dir = data_cfg.get("preprocessed_dir")

    datasets: List = []
    for modality in modalities:
        for source_field in fields:
            for target_field in fields:
                if source_field == target_field:
                    continue
                try:
                    if preprocessed_dir:
                        base = CachedPairedDataset(
                            preprocessed_dir=preprocessed_dir,
                            split=split,
                            modality=modality,
                            source_field=source_field,
                            target_field=target_field,
                            crop_size=crop_size,
                        )
                    else:
                        base = PairedMRIDataset(
                            data_root=data_cfg["data_dir"],
                            split=SPLIT_MAP.get(split, split),
                            modality=modality,
                            source_field=source_field,
                            target_field=target_field,
                            crop_size=crop_size,
                            slice_axis=data_cfg.get("slice_axis", 2),
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


def train_task3_conditioned_finetune(
    cfg: dict,
    device: torch.device,
    use_dist: bool = False,
    resume_ckpt: str | None = None,
) -> None:
    """Supervised prospective fine-tuning over every ordered field pair."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    pretrain_cfg = cfg.get("pretrain", {})
    finetune_cfg = cfg.get("finetune", {})
    _, world_size = get_dist_info()

    output_dir = Path(train_cfg["output_dir"])
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    generator = build_generator(model_cfg, data_cfg=data_cfg, task=3).to(device)
    pretrained = pretrain_cfg.get("checkpoint")
    if pretrained:
        state = torch.load(pretrained, map_location=device, weights_only=False)
        generator.load_state_dict(_extract_generator_state(state))
        if is_main_process():
            print(f"Loaded Task 3 UNETR pretraining from: {pretrained}")
    else:
        initialize_generator(generator)
        if is_main_process():
            print("Task 3 UNETR fine-tuning from random initialization")

    if use_dist:
        generator = DDP(generator, device_ids=[device.index])

    dataset = _paired_task3_dataset(cfg)
    batch_size = int(train_cfg.get("batch_size", 1))
    if use_dist:
        batch_size = max(1, batch_size // world_size)
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        persistent_workers=train_cfg.get("num_workers", 4) > 0,
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

    start_epoch = 0
    if resume_ckpt:
        checkpoint = torch.load(
            resume_ckpt, map_location=device, weights_only=False
        )
        network = generator.module if use_dist else generator
        network.load_state_dict(checkpoint.get("generator", checkpoint))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))

    epochs = int(train_cfg.get("epochs", 100))
    max_iters = int(train_cfg.get("max_iters_per_epoch", 0) or 0)
    steps_per_epoch = min(len(loader), max_iters) if max_iters else len(loader)
    global_step = start_epoch * max(steps_per_epoch, 1)

    if is_main_process():
        print(
            f"Task 3 paired UNETR fine-tuning: epochs={start_epoch + 1}-{epochs}, "
            f"pairs={len(dataset)}, iters/epoch={len(loader)}"
            + (f" capped_at={max_iters}" if max_iters else "")
        )

    for epoch in range(start_epoch, epochs):
        generator.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total = 0.0
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
            source_ids, target_ids, modality_ids = _conditions(batch, device)
            task_ids = torch.full(
                (source.shape[0],), 2, dtype=torch.long, device=device
            )
            prediction = generator(
                source,
                source_field_id=source_ids,
                target_field_id=target_ids,
                modality_id=modality_ids,
                task_id=task_ids,
            )

            loss_l1 = criterion_l1(prediction, target)
            loss_lpips = criterion_lpips(prediction, target)
            loss_ssim = (
                criterion_ssim(prediction, target)
                if lambda_ssim > 0
                else torch.zeros((), device=device)
            )
            loss_edge = (
                criterion_edge(prediction, target)
                if criterion_edge is not None
                else torch.zeros((), device=device)
            )
            loss = (
                lambda_l1 * loss_l1
                + lambda_lpips * loss_lpips
                + lambda_ssim * loss_ssim
                + lambda_edge * loss_edge
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += float(loss.detach())
            completed += 1
            global_step += 1
            if is_main_process():
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
                if should_log(global_step):
                    payload = {
                        "task3_finetune/loss_total": float(loss.detach()),
                        "task3_finetune/loss_l1": float(loss_l1.detach()),
                        "task3_finetune/loss_lpips": float(loss_lpips.detach()),
                        "task3_finetune/loss_ssim": float(loss_ssim.detach()),
                        "task3_finetune/loss_edge": float(loss_edge.detach()),
                        "task3_finetune/lr": optimizer.param_groups[0]["lr"],
                        "task3_finetune/epoch": epoch + 1,
                    }
                    if should_log_images(global_step):
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
        if is_main_process():
            print(f"Task3 finetune epoch {epoch + 1}: avg_loss={average:.6f}")
            log_wandb(
                {
                    "task3_finetune_epoch/avg_loss": average,
                    "task3_finetune_epoch/index": epoch + 1,
                },
                step=global_step,
            )
            update_summary({
                "last_task3_finetune_epoch": epoch + 1,
                "last_task3_finetune_loss": average,
            })

        if is_main_process() and (epoch + 1) % int(train_cfg.get("save_every", 10)) == 0:
            path = weights_dir / f"checkpoint_epoch{epoch + 1}.pth"
            network = generator.module if use_dist else generator
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )
            print(f"Saved: {path}")
            update_summary({"latest_checkpoint": str(path)})
