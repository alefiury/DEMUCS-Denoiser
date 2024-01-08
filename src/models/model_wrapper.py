from typing import Dict, Union

import torch
import wandb
import lightning as pl
from lion_pytorch import Lion
from torch.optim import Adam, AdamW
from torch.optim.optimizer import Optimizer

from utils.loss import loss_function
from models.model import DemucsDenoiser
from utils.scheduler import CosineWarmupLR, LinearLR
from utils.data_augmentation import BandMask, Remix, Shift, RevEcho

class DemucsDenoiserWrapper(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.loss_function = loss_function
        self.automatic_optimization = False
        self.denoiser = DemucsDenoiser(
            **config["model"],
            training_sample_rate=config.data["sampling_rate"]
        )

        # data augmentation
        augments = []
        if config.data_augmentation.remix:
            augments.append(Remix())
        if config.data_augmentation.bandmask:
            augments.append(BandMask(config.data_augmentation.bandmask, sample_rate=config.data.sampling_rate))
        if config.data_augmentation.shift:
            augments.append(Shift(config.data_augmentation.shift, config.data_augmentation.shift_same))
        if config.data_augmentation.revecho:
            augments.append(
                RevEcho(config.data_augmentation.revecho))
        self.augment = torch.nn.Sequential(*augments)

    def configure_optimizers(self) -> Union[Optimizer, Dict]:
        """Configures the optimizer and the learning rate scheduler."""
        opt_params = self.config.optimizer["params"]
        scheduler_params = self.config.scheduler["params"]

        if self.config.optimizer.name.lower() == "adam":
            optimizer = Adam(
                self.parameters(),
                eps=opt_params["eps"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"]
            )

        elif self.config.optimizer.name.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(),
                eps=opt_params["eps"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"]
            )

        elif self.config.optimizer.name.lower() == "lion":
            optimizer = Lion(
                self.parameters(),
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"],
                use_triton=opt_params.get("use_triton", False),
            )

        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer.name}")

        if not self.config["scheduler"]:
            return optimizer

        scheduler = None
        if self.config.scheduler.name.lower() == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=scheduler_params.get("patience", self.trainer.max_steps*0.25),
                factor=0.9,
                min_lr=opt_params.get("min_learning_rate", 1.0e-6)
            )

        if self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=opt_params.get("min_learning_rate", 1.0e-6),
                lr_max=opt_params["learning_rate"],
                warmup=scheduler_params.get("warmup_lr", self.trainer.max_steps*0.05),
                T_max=self.trainer.max_steps
            )

        if self.config.scheduler.name.lower() == "linearlr":
            scheduler = LinearLR(
                optimizer,
                start_factor=scheduler_params.get("start_factor", 1.0 / 3.0),
                end_factor=scheduler_params.get("end_factor", 1.0),
                total_iters=scheduler_params.get("total_iters", 5),
                last_epoch=scheduler_params.get("last_epoch", -1),
                verbose=scheduler_params.get("verbose", False)
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Forward pass for the model."""
        return self.denoiser(noisy)

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> None:
        """Training step."""
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        noisy, target = train_batch

        sources = torch.stack([noisy - target, target])
        sources = self.augment(sources)
        noise, target = sources
        noisy = noise + target

        estimate = self(noisy)

        total_loss, mrstft_loss, l1_loss = self.loss_function(estimate, target)

        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagate the loss
        self.manual_backward(total_loss)
        # Update the weights
        optimizer.step()
        # Update the learning rate
        if lr_scheduler:
            # If scheduler requires loss value (e.g., ReduceLROnPlateau), pass the average loss over accumulated steps
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(total_loss.detach().item())
            else:
                lr_scheduler.step()

        self.log("train-total_loss", total_loss.detach().item())
        self.log("train-mrstft_loss", mrstft_loss.detach().item())
        self.log("train-l1_loss", l1_loss.detach().item())

        if batch_idx % self.config.logging["log_audio_every"] == 0:
            audio_examples = []
            audio_examples.append(wandb.Audio(target[0].squeeze(0).detach().cpu().numpy(), caption='target_wav', sample_rate=self.config.data.sampling_rate))
            audio_examples.append(wandb.Audio(noisy[0].squeeze(0).detach().cpu().numpy(), caption='noisy_wav', sample_rate=self.config.data.sampling_rate))
            audio_examples.append(wandb.Audio(estimate[0].squeeze(0).detach().cpu().numpy(), caption='estimate_wav', sample_rate=self.config.data.sampling_rate))
            self.logger.experiment.log({
                "train-audios": audio_examples
            })

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> None:
        noisy, target = val_batch
        estimate = self(noisy)

        total_loss, mrstft_loss, l1_loss = self.loss_function(estimate, target)

        self.log("val-total_loss", total_loss.detach().item())
        self.log("val-mrstft_loss", mrstft_loss.detach().item())
        self.log("val-l1_loss", l1_loss.detach().item())

        # Log audios only on the last validation step
        if batch_idx == len(self.val_dataloader()) - 1:
            audio_examples = []
            audio_examples.append(wandb.Audio(target[0].squeeze(0).cpu().numpy(), caption='target_wav', sample_rate=self.config.data.sampling_rate))
            audio_examples.append(wandb.Audio(noisy[0].squeeze(0).cpu().numpy(), caption='noisy_wav', sample_rate=self.config.data.sampling_rate))
            audio_examples.append(wandb.Audio(estimate[0].squeeze(0).cpu().numpy(), caption='estimate_wav', sample_rate=self.config.data.sampling_rate))
            self.logger.experiment.log({
                "val-audios": audio_examples
            })