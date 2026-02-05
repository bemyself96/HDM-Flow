import os
import json
import random
import logging
import argparse
import warnings
import math
import numpy as np
from glob import glob
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from lora import inject_lora
from models import DiT_models
from dataloader import Data_Loader
from utils import backup_file, build_scheduler

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def settings(seed):
    """
    Fixed seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    logger = logging.getLogger(__name__)

    return logger


def main():

    # Parameters for training:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--image_size", type=list, default=256)
    parser.add_argument("--learn_rate", type=float, default=1e-4)
    parser.add_argument("--global_seed", type=int, default=114514)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument("--model", type=str, default="DiT-S/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    parser.add_argument("--data_dme_path_json", type=str, default="./datasets/train.json")
    parser.add_argument("--data_cdme_path", type=str, default="/home/Data/Work3/h5_data/CDMEnp")
    parser.add_argument("--data_sdme_path", type=str, default="/home/Data/Work3/h5_data/SDME")
    parser.add_argument("--results_dir", type=str, default="/home/Data/Work3/models/FMTA")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--fine_tune", type=str, default=True)
    parser.add_argument("--pre_model_path", type=str, default="/home/Data/Pre_models")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/Data/Work3/models/FM_Pre/002-DiT-S-2/checkpoints/0510000.pt",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    settings(args.global_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    save_image_dir = f"{experiment_dir}/images"
    backup_file_dir = f"{experiment_dir}/backup"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(backup_file_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    backup_file(backup_file_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](input_size=latent_size, in_channels=args.in_channels)

    # Load base model:
    if args.fine_tune:
        dit_ckpt = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(dit_ckpt["ema"], strict=False)

    # injection Lora  and freeze all layers except the finetune layer
    for name, param in model.named_modules():
        name_list = name.split(".")
        filter_list = ["qkv"]
        if any(f in name_list for f in filter_list) and isinstance(param, nn.Linear):
            inject_lora(model, name, param, args.lora_rank, args.lora_alpha)

    for name, param in model.named_parameters():
        if name.endswith("lora_a") or name.endswith("lora_b") or "finenet" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(device)
    # logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    with open(args.data_dme_path_json, "r") as f:
        img_list = json.load(f)

    train_dataset = Data_Loader(args.data_sdme_path, img_list, args.data_cdme_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images.")

    # Setup loss function
    mse_loss = nn.MSELoss().to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learn_rate, betas=(0.9, 0.95), weight_decay=0)
    steps_per_epoch = len(train_dataset) // args.batch_size
    warmup_steps = int(args.epochs * steps_per_epoch * 0.05)  # 5%
    scheduler = build_scheduler(optimizer, warmup_steps, args.epochs * steps_per_epoch)

    log_steps = 0
    running_loss = 0
    train_steps = 0

    logger.info(f"Training for {args.epochs} epochs ...")
    start_time = time()

    for epoch in range(args.epochs):
        model.train()

        logger.info(f"Beginning epoch {epoch} ...")

        for sample in train_loader:

            alpha = sample["alpha"].to(device)  # B
            alpha = alpha.float()
            delta_t23 = sample["t23"].to(device)

            vae_t1 = sample["vae_t1"].to(device)  # B, 4, 32, 32
            vae_t2 = sample["vae_t2"].to(device)
            vae_t3 = sample["vae_t3"].to(device)

            v_his = vae_t2 - vae_t1

            t = torch.rand(vae_t1.shape[0]).to(device)

            noise = torch.randn_like(vae_t1).to(device)
            img_t = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * vae_t3

            pred_v = model(img_t, t, vae_t2, v_his, alpha, delta_t23)  # B, 8, 32, 32

            loss = mse_loss(pred_v, vae_t3 - noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                logger.info(
                    f"(Epoch={epoch:04d}, step={train_steps:07d}) Train Loss: MSE = {avg_loss:.6f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                log_steps = 0
                running_loss = 0

                start_time = time()

        # Save DiT checkpoint:
        if (epoch + 1) % 5 == 0:
            lora_state_dict = {}
            finenet_state_dict = {}
            for name, param in model.named_parameters():
                if name.endswith("lora_a") or name.endswith("lora_b"):
                    lora_state_dict[name] = param
                if "_finenet" in name:
                    finenet_state_dict[name] = param
            dit_ckpt = {
                "lora": lora_state_dict,
                "finenet": finenet_state_dict,
            }

            checkpoint_path = f"{checkpoint_dir}/{epoch+1:04d}.pt"
            torch.save(dit_ckpt, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training Done!")


if __name__ == "__main__":
    main()
