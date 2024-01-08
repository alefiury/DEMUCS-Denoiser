import os
import argparse

import torch
import wandb
import lightning as pl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataloader import ChuncksDataset
from data.utils import build_metadata_valentini
from models.model_wrapper import DemucsDenoiserWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", type=str, required=True)
    parser.add_argument("-np", "--noisy-path", type=str, required=True)
    parser.add_argument("-cp", "--clean-path", type=str, required=True)
    parser.add_argument("-g", "--gpu", type=int, required=True)
    parser.add_argument(
        "-ck",
        "--checkpoint-dir",
        required=False,
        type=str,
        default="../checkpoints/demucs"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    noisy_data_dir = args.noisy_path
    clean_data_dir = args.clean_path

    train_filenames, val_filenames = build_metadata_valentini(
        noisy_data_dir=noisy_data_dir,
        clean_data_dir=clean_data_dir,
        speakers_val="p286,p287"
    )

    print(f"Number of training examples: {len(train_filenames)}")
    print(f"Number of validation examples: {len(val_filenames)}")

    true_max_length = config.data["max_length"] * config.data["sampling_rate"]
    true_stride = config.data["stride"] * config.data["sampling_rate"]

    train_dataset = ChuncksDataset(
        filenames=train_filenames,
        max_length=true_max_length,
        stride=true_stride,
        pad=config.data["pad"],
        noisy_base_dir=noisy_data_dir,
        clean_base_dir=clean_data_dir
    )

    val_dataset = ChuncksDataset(
        filenames=val_filenames,
        noisy_base_dir=noisy_data_dir,
        clean_base_dir=clean_data_dir
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train["batch_size"],
        shuffle=config.train["shuffle"],
        num_workers=config.train["num_workers"],
        drop_last=config.train["drop_last"]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train["batch_size"],
        num_workers=config.train["num_workers"],
        shuffle=False,
        drop_last=False
    )

    model = DemucsDenoiserWrapper(config)

    wandb.init(project="Demucs", name=config.title, entity="alefiury")
    logger = WandbLogger(project="Demucs", name=config.title, entity="alefiury")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
        default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()