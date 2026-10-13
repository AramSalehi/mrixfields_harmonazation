"""CUT (Contrastive Unpaired Translation) model.

The standard ResNet generator remains the default. A conditioned 2D UNETR can
be selected through ``model.name: conditioned_unetr2d`` in the YAML config.
Both generators use the same dataset, discriminator, PatchNCE loss, optimizer,
scheduler, checkpoint, DDP, and fine-tuning pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .generator_factory import (
    build_generator,
    forward_generator,
    get_generator_name,
    initialize_generator,
    is_conditioned_generator,
)
from .networks import NLayerDiscriminator, PatchSampleF, init_net
from ..losses.patchnce import PatchNCELoss
from ..losses.adversarial import GANLoss


class CUTModel(nn.Module):
    """CUT model with a selectable ResNet or conditioned-UNETR generator."""

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        ndf=64,
        n_blocks=9,
        nce_layers=None,
        nce_T=0.07,
        num_patches=256,
        lambda_GAN=1.0,
        lambda_NCE=1.0,
        nce_idt=True,
        lambda_identity=0.0,
        gan_mode="lsgan",
        netF_nc=256,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        generator_config=None,
        data_config=None,
        task=1,
    ):
        super().__init__()

        if nce_layers is None:
            nce_layers = [0, 4, 8, 12, 16]
        self.nce_layers = list(nce_layers)
        self.nce_T = nce_T
        self.num_patches = num_patches
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.nce_idt = nce_idt
        self.lambda_identity = lambda_identity
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.task = int(task)

        generator_config = dict(generator_config or {})
        generator_config.setdefault("input_nc", input_nc)
        generator_config.setdefault("output_nc", output_nc)
        generator_config.setdefault("ngf", ngf)
        generator_config.setdefault("n_blocks", n_blocks)
        self.data_config = dict(data_config or {})
        self.generator_name = get_generator_name(generator_config)

        self.source_field = self.data_config.get("source_field", "0.1T")
        self.target_field = self.data_config.get("target_field", "7T")
        modalities = self.data_config.get("modalities") or ["T1W"]
        self.modality = modalities[0]

        self.netG = build_generator(
            generator_config,
            data_cfg=self.data_config,
            task=self.task,
        )
        self.netD = NLayerDiscriminator(output_nc, ndf)
        self.netF = PatchSampleF(use_mlp=True, nc=netF_nc)

        self.criterionGAN = GANLoss(mode=gan_mode)
        self.criterionNCE = nn.ModuleList(
            [PatchNCELoss(nce_T=nce_T) for _ in self.nce_layers]
        )
        self.criterionIdentity = nn.L1Loss()

    @property
    def uses_conditioned_generator(self):
        return is_conditioned_generator(self.netG)

    def init_weights(self, init_type="normal", init_gain=0.02):
        """Initialize networks while preserving UNETR AdaLN-Zero weights."""
        initialize_generator(self.netG, init_type, init_gain)
        init_net(self.netD, init_type, init_gain)

    def setup_optimizers(self):
        """Create G and D optimizers after the PatchSampleF lazy init."""
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

    def _generator_forward(
        self,
        image,
        source_field,
        target_field,
        layers=None,
        encode_only=False,
    ):
        return forward_generator(
            self.netG,
            image,
            source_field=source_field,
            target_field=target_field,
            modality=self.modality,
            task=self.task,
            layers=layers,
            encode_only=encode_only,
        )

    def data_dependent_initialize(self, real_A, real_B):
        """Create the lazy PatchSampleF MLP heads from real features."""
        if self.uses_conditioned_generator:
            # The original CUT code uses preliminary backward passes solely to
            # trigger lazy MLP creation. For a transformer this is needlessly
            # memory-intensive, so create the same MLPs directly from features.
            with torch.no_grad():
                features = self._generator_forward(
                    real_A,
                    self.source_field,
                    self.target_field,
                    layers=self.nce_layers,
                    encode_only=True,
                )
            self.netF(features, self.num_patches, None)
        else:
            # Preserve the baseline ResNet initialization behavior exactly.
            self.forward(real_A, real_B)
            self.compute_D_loss(real_A, real_B).backward()
            self.compute_G_loss(real_A, real_B).backward()

        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

    def forward(self, real_A, real_B):
        """Generate source-to-target and optional target identity outputs."""
        if self.nce_idt and self.training:
            if self.uses_conditioned_generator:
                batch_a = real_A.shape[0]
                real = torch.cat((real_A, real_B), dim=0)

                # First half: source -> target translation.
                # Second half: target -> target identity mapping.
                source_ids = torch.cat(
                    [
                        torch.full(
                            (batch_a,),
                            self.netG.module.default_source_field_id
                            if hasattr(self.netG, "module")
                            else self.netG.default_source_field_id,
                            dtype=torch.long,
                            device=real.device,
                        ),
                        torch.full(
                            (real_B.shape[0],),
                            self.netG.module.default_target_field_id
                            if hasattr(self.netG, "module")
                            else self.netG.default_target_field_id,
                            dtype=torch.long,
                            device=real.device,
                        ),
                    ]
                )
                target_id = (
                    self.netG.module.default_target_field_id
                    if hasattr(self.netG, "module")
                    else self.netG.default_target_field_id
                )
                modality_id = (
                    self.netG.module.default_modality_id
                    if hasattr(self.netG, "module")
                    else self.netG.default_modality_id
                )
                task_id = (
                    self.netG.module.default_task_id
                    if hasattr(self.netG, "module")
                    else self.netG.default_task_id
                )
                target_ids = torch.full(
                    (real.shape[0],),
                    target_id,
                    dtype=torch.long,
                    device=real.device,
                )
                modality_ids = torch.full(
                    (real.shape[0],),
                    modality_id,
                    dtype=torch.long,
                    device=real.device,
                )
                task_ids = torch.full(
                    (real.shape[0],),
                    task_id,
                    dtype=torch.long,
                    device=real.device,
                )
                fake = self.netG(
                    real,
                    source_field_id=source_ids,
                    target_field_id=target_ids,
                    modality_id=modality_ids,
                    task_id=task_ids,
                )
            else:
                real = torch.cat((real_A, real_B), dim=0)
                fake = self.netG(real)

            self.fake_B = fake[: real_A.size(0)]
            self.idt_B = fake[real_A.size(0) :]
        else:
            self.fake_B = self._generator_forward(
                real_A,
                self.source_field,
                self.target_field,
            )
        return self.fake_B

    def compute_D_loss(self, real_A, real_B):
        """Calculate the unchanged CUT PatchGAN discriminator loss."""
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        pred_real = self.netD(real_B)
        loss_D_real = self.criterionGAN(pred_real, True).mean()
        return (loss_D_fake + loss_D_real) * 0.5

    def compute_G_loss(self, real_A, real_B):
        """Calculate GAN + PatchNCE + optional identity reconstruction loss."""
        fake = self.fake_B

        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            loss_G_GAN = (
                self.criterionGAN(pred_fake, True).mean()
                * self.lambda_GAN
            )
        else:
            loss_G_GAN = 0.0

        if self.lambda_NCE > 0.0:
            loss_NCE = self.calculate_NCE_loss(
                real_A,
                self.fake_B,
                source_field=self.source_field,
                target_field=self.target_field,
            )
        else:
            loss_NCE = 0.0

        if self.nce_idt and self.lambda_NCE > 0.0 and self.training:
            loss_NCE_Y = self.calculate_NCE_loss(
                real_B,
                self.idt_B,
                source_field=self.target_field,
                target_field=self.target_field,
            )
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        loss_identity = 0.0
        if (
            self.nce_idt
            and self.lambda_identity > 0.0
            and self.training
        ):
            loss_identity = (
                self.criterionIdentity(self.idt_B, real_B)
                * self.lambda_identity
            )

        return loss_G_GAN + loss_NCE_both + loss_identity

    def calculate_NCE_loss(
        self,
        src,
        tgt,
        source_field,
        target_field,
    ):
        """Calculate PatchNCE using identical conditions for both encodings."""
        n_layers = len(self.nce_layers)
        feat_q = self._generator_forward(
            tgt,
            source_field,
            target_field,
            layers=self.nce_layers,
            encode_only=True,
        )
        feat_k = self._generator_forward(
            src,
            source_field,
            target_field,
            layers=self.nce_layers,
            encode_only=True,
        )

        feat_k_pool, sample_ids = self.netF(
            feat_k, self.num_patches, None
        )
        feat_q_pool, _ = self.netF(
            feat_q, self.num_patches, sample_ids
        )

        total_nce_loss = 0.0
        for f_q, f_k, criterion in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE
        ):
            loss = criterion(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def optimize_parameters(self, real_A, real_B):
        """Run the existing CUT D step followed by G and F steps."""
        self.forward(real_A, real_B)

        for parameter in self.netD.parameters():
            parameter.requires_grad = True
        self.optimizer_D.zero_grad()
        loss_D = self.compute_D_loss(real_A, real_B)
        loss_D.backward()
        self.optimizer_D.step()

        for parameter in self.netD.parameters():
            parameter.requires_grad = False
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        loss_G = self.compute_G_loss(real_A, real_B)
        loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_F.step()

        return {"loss_D": loss_D.item(), "loss_G": loss_G.item()}
