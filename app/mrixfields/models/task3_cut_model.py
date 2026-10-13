"""Multi-domain CUT objective for the conditioned 2D UNETR (Task 3).

The generator is conditioned on source field, target field, modality, and task
inside every ViT and decoder block.  Retrospective source/target images are
unpaired.  A target-conditional PatchGAN distinguishes real and generated
images for the requested field and modality, while PatchNCE preserves anatomy.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioned_unetr import FIELD_TO_ID, MODALITY_TO_ID
from .generator_factory import (
    build_generator, initialize_generator, is_conditioned_generator,
)
from .networks import NLayerDiscriminator, PatchSampleF, init_net
from ..losses.adversarial import GANLoss
from ..losses.patchnce import PatchNCELoss


def condition_ids(
    values,
    mapping: dict,
    device: torch.device,
) -> torch.Tensor:
    """Convert collated strings, integer lists, or tensors to long IDs."""
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.long)
    if isinstance(values, str):
        values = [values]
    ids = []
    for value in values:
        if isinstance(value, str):
            if value not in mapping:
                raise ValueError(
                    f"Unknown condition {value!r}; expected {list(mapping)}."
                )
            ids.append(mapping[value])
        else:
            ids.append(int(value))
    return torch.tensor(ids, dtype=torch.long, device=device)


class ConditionalPatchDiscriminator(nn.Module):
    """PatchGAN conditioned by one-hot target field and modality maps."""

    def __init__(
        self,
        image_channels: int = 1,
        ndf: int = 64,
        num_fields: int = 5,
        num_modalities: int = 3,
    ) -> None:
        super().__init__()
        self.num_fields = int(num_fields)
        self.num_modalities = int(num_modalities)
        self.discriminator = NLayerDiscriminator(
            input_nc=image_channels + num_fields + num_modalities,
            ndf=ndf,
        )

    def forward(
        self,
        image: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = image.shape
        field = F.one_hot(
            target_field_id, num_classes=self.num_fields
        ).to(dtype=image.dtype)
        modality = F.one_hot(
            modality_id, num_classes=self.num_modalities
        ).to(dtype=image.dtype)
        condition = torch.cat([field, modality], dim=1)
        condition = condition[:, :, None, None].expand(
            batch, condition.shape[1], height, width
        )
        return self.discriminator(torch.cat([image, condition], dim=1))


class Task3ConditionedCUTModel(nn.Module):
    """CUT with a single any-field-to-any-field conditioned UNETR generator."""

    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        *,
        lr: float = 2e-4,
        discriminator_lr: float | None = None,
        gradient_clip_norm: float = 0.0,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ) -> None:
        super().__init__()
        self.model_config = dict(model_config)
        self.data_config = dict(data_config)
        self.nce_layers = list(
            self.model_config.get("nce_layers", [2, 5, 8, 11])
        )
        self.num_patches = int(self.model_config.get("num_patches", 256))
        self.lambda_GAN = float(self.model_config.get("lambda_GAN", 1.0))
        self.lambda_NCE = float(self.model_config.get("lambda_NCE", 1.0))
        self.lambda_identity = float(
            self.model_config.get("lambda_identity", 1.0)
        )
        self.nce_idt = bool(self.model_config.get("nce_idt", True))
        self.lr = float(lr)
        self.discriminator_lr = (
            float(discriminator_lr) if discriminator_lr is not None else self.lr
        )
        self.gradient_clip_norm = float(gradient_clip_norm or 0.0)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

        self.netG = build_generator(
            self.model_config,
            data_cfg=self.data_config,
            task=3,
        )
        if not is_conditioned_generator(self.netG):
            raise ValueError(
                "Task3ConditionedCUTModel requires a conditioned generator; "
                f"the factory built {type(self.netG).__name__}."
            )
        self.netD = ConditionalPatchDiscriminator(
            image_channels=self.model_config.get("output_nc", 1),
            ndf=self.model_config.get("ndf", 64),
        )
        self.netF = PatchSampleF(
            use_mlp=True,
            nc=self.model_config.get("netF_nc", 256),
        )

        self.criterionGAN = GANLoss(
            mode=self.model_config.get("gan_mode", "lsgan")
        )
        self.criterionNCE = nn.ModuleList(
            [
                PatchNCELoss(
                    nce_T=self.model_config.get("nce_T", 0.07)
                )
                for _ in self.nce_layers
            ]
        )
        self.criterionIdentity = nn.L1Loss()

    def init_weights(self) -> None:
        initialize_generator(self.netG)
        init_net(self.netD)

    def setup_optimizers(self) -> None:
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(),
            lr=self.discriminator_lr,
            betas=(self.beta1, self.beta2),
        )

    @staticmethod
    def _task_ids(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size,), 2, dtype=torch.long, device=device
        )

    def _generator(
        self,
        image: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
        *,
        layers: Sequence[int] | None = None,
        encode_only: bool = False,
    ):
        return self.netG(
            image,
            list(layers or []),
            encode_only,
            source_field_id=source_field_id,
            target_field_id=target_field_id,
            modality_id=modality_id,
            task_id=self._task_ids(image.shape[0], image.device),
        )

    def data_dependent_initialize(
        self,
        source: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            features = self._generator(
                source,
                source_field_id,
                target_field_id,
                modality_id,
                layers=self.nce_layers,
                encode_only=True,
            )
        self.netF(features, self.num_patches, None)
        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.nce_idt and self.training:
            combined = torch.cat([source, target], dim=0)
            combined_source = torch.cat(
                [source_field_id, target_field_id], dim=0
            )
            combined_target = torch.cat(
                [target_field_id, target_field_id], dim=0
            )
            combined_modality = torch.cat(
                [modality_id, modality_id], dim=0
            )
            output = self._generator(
                combined,
                combined_source,
                combined_target,
                combined_modality,
            )
            self.fake_target = output[: source.shape[0]]
            self.identity_target = output[source.shape[0] :]
        else:
            self.fake_target = self._generator(
                source,
                source_field_id,
                target_field_id,
                modality_id,
            )
        return self.fake_target

    def compute_D_loss(
        self,
        target: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> torch.Tensor:
        prediction_fake = self.netD(
            self.fake_target.detach(), target_field_id, modality_id
        )
        prediction_real = self.netD(
            target, target_field_id, modality_id
        )
        loss_fake = self.criterionGAN(prediction_fake, False).mean()
        loss_real = self.criterionGAN(prediction_real, True).mean()
        return 0.5 * (loss_fake + loss_real)

    def calculate_NCE_loss(
        self,
        source: torch.Tensor,
        generated: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> torch.Tensor:
        features_query = self._generator(
            generated,
            source_field_id,
            target_field_id,
            modality_id,
            layers=self.nce_layers,
            encode_only=True,
        )
        features_key = self._generator(
            source,
            source_field_id,
            target_field_id,
            modality_id,
            layers=self.nce_layers,
            encode_only=True,
        )
        pooled_key, sample_ids = self.netF(
            features_key, self.num_patches, None
        )
        pooled_query, _ = self.netF(
            features_query, self.num_patches, sample_ids
        )

        total = 0.0
        for query, key, criterion in zip(
            pooled_query, pooled_key, self.criterionNCE
        ):
            total = total + criterion(query, key).mean()
        return self.lambda_NCE * total / max(len(self.nce_layers), 1)

    def compute_G_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> torch.Tensor:
        loss_gan = torch.zeros((), device=source.device)
        if self.lambda_GAN > 0:
            prediction_fake = self.netD(
                self.fake_target, target_field_id, modality_id
            )
            loss_gan = (
                self.criterionGAN(prediction_fake, True).mean()
                * self.lambda_GAN
            )

        loss_nce = torch.zeros((), device=source.device)
        if self.lambda_NCE > 0:
            loss_nce = self.calculate_NCE_loss(
                source,
                self.fake_target,
                source_field_id,
                target_field_id,
                modality_id,
            )

        if self.nce_idt and self.lambda_NCE > 0 and self.training:
            identity_nce = self.calculate_NCE_loss(
                target,
                self.identity_target,
                target_field_id,
                target_field_id,
                modality_id,
            )
            loss_nce = 0.5 * (loss_nce + identity_nce)

        loss_identity = torch.zeros((), device=source.device)
        if self.nce_idt and self.lambda_identity > 0 and self.training:
            loss_identity = (
                self.criterionIdentity(self.identity_target, target)
                * self.lambda_identity
            )

        self.loss_G_GAN = loss_gan.detach()
        self.loss_NCE = loss_nce.detach()
        self.loss_identity = loss_identity.detach()
        return loss_gan + loss_nce + loss_identity

    def optimize_parameters(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_field_id: torch.Tensor,
        target_field_id: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> dict:
        self.forward(
            source,
            target,
            source_field_id,
            target_field_id,
            modality_id,
        )

        for parameter in self.netD.parameters():
            parameter.requires_grad = True
        self.optimizer_D.zero_grad(set_to_none=True)
        loss_D = self.compute_D_loss(
            target, target_field_id, modality_id
        )
        loss_D.backward()
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.netD.parameters(), self.gradient_clip_norm
            )
        self.optimizer_D.step()

        for parameter in self.netD.parameters():
            parameter.requires_grad = False
        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_F.zero_grad(set_to_none=True)
        loss_G = self.compute_G_loss(
            source,
            target,
            source_field_id,
            target_field_id,
            modality_id,
        )
        loss_G.backward()
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.netG.parameters(), self.gradient_clip_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.netF.parameters(), self.gradient_clip_norm
            )
        self.optimizer_G.step()
        self.optimizer_F.step()

        return {
            "loss_D": float(loss_D.detach()),
            "loss_G": float(loss_G.detach()),
            "loss_G_GAN": float(self.loss_G_GAN),
            "loss_NCE": float(self.loss_NCE),
            "loss_identity": float(self.loss_identity),
        }
