import os
import cv2
import json
import natsort
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models import DiT_models
from dataloader import Data_Loader
from diffusers.models import AutoencoderKL
from lora import inject_lora, LoraLayer

warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--model", type=str, default="DiT-S/2")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)

    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--data_dme_path_json", type=str, default="./datasets/test.json")
    parser.add_argument("--data_cdme_path", type=str, default="/home/Data/Work3/h5_data/CDME")
    parser.add_argument("--data_sdme_path", type=str, default="/home/Data/Work3/h5_data/SDME")
    parser.add_argument("--pre_model_path", type=str, default="/home/Data/Pre_models")
    parser.add_argument(
        "--ckpt_main",
        type=str,
        default="/home/Data/Work3/models/FM_Pre/002-DiT-S-2/checkpoints/",
    )
    parser.add_argument(
        "--ckpt_sub",
        type=str,
        default="/home/Data/Work3/models/FMTA/002-DiT-S-2/checkpoints/",
    )
    args = parser.parse_args()

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_exp_name = args.ckpt_sub.split("/")[-3]

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, in_channels=args.in_channels)

    main_checkpoint_name = natsort.natsorted(os.listdir(args.ckpt_main))[-1]  # latest_ckpt.pt
    sub_checkpoint_name = natsort.natsorted(os.listdir(args.ckpt_sub))[-1]

    checkpoint_main_path = os.path.join(args.ckpt_main, main_checkpoint_name)
    print(f"Loading main checkpoint from {checkpoint_main_path}")

    checkpoint_sub_path = os.path.join(args.ckpt_sub, sub_checkpoint_name)
    print(f"Loading sub checkpoint from {checkpoint_sub_path}")

    checkpoint_main = torch.load(checkpoint_main_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint_main["ema"], strict=False)

    # inject lora layer into the model
    for name, param in model.named_modules():
        name_list = name.split(".")
        filter_list = ["qkv"]
        if any(f in name_list for f in filter_list) and isinstance(param, nn.Linear):
            inject_lora(model, name, param, args.lora_rank, args.lora_alpha)

    checkpoint_sub = torch.load(checkpoint_sub_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint_sub["lora"], strict=False)
    model.load_state_dict(checkpoint_sub["finenet"], strict=False)
    model.to(device)

    # intergrate the lora layer into the model
    for name, param in model.named_modules():
        name_list = name.split(".")
        if isinstance(param, LoraLayer):
            children = name_list[:-1]
            cur_layer = model
            for child in children:
                cur_layer = getattr(cur_layer, child)
            lora_weight = (param.lora_a @ param.lora_b) * param.alpha / param.rank
            param.raw_layer.weight = nn.Parameter(param.raw_layer.weight.add(lora_weight.T)).to(device)
            setattr(cur_layer, name_list[-1], param.raw_layer)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f"{args.pre_model_path}/sd-vae-ft-{args.vae}").to(device)

    with open(args.data_dme_path_json, "r") as f:
        img_list = json.load(f)
    train_dataset = Data_Loader(args.data_sdme_path, img_list, args.data_cdme_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Create sampling noise:

    # z = torch.randn(args.batch_size, 4, latent_size, latent_size, device=device)

    with torch.no_grad():
        for batch, sample in enumerate(train_loader):

            alpha = sample["alpha"].to(device)  # B
            alpha = alpha.float()
            delta_t23 = sample["t23"].to(device)

            vae_t1 = sample["vae_t1"].to(device)  # B, 4, 32, 32
            vae_t2 = sample["vae_t2"].to(device)
            # vae_t3 = sample["vae_t3"].to(device)

            v_his = vae_t2 - vae_t1

            Bs = vae_t1.shape[0]

            filename = sample["name"]
            ft1 = sample["ft1"]  # W0
            ft2 = sample["ft2"]  # W4
            ft3 = sample["ft3"]  # W8

            z = torch.randn(Bs, 4, latent_size, latent_size, device=device)

            for i in tqdm(range(args.num_sampling_steps)):
                t = torch.tensor(1.0 / args.num_sampling_steps * i).expand(Bs).to(device)

                pred_v = model(z, t, vae_t2, v_his, alpha, delta_t23)  # B,8,32,32
                z = z + pred_v * 1.0 / args.num_sampling_steps
                z = z.detach()

            samples = vae.decode(z / 0.18215).sample
            samples = samples[:, 0, :, :].unsqueeze(1)

            for ib in range(vae_t1.shape[0]):
                fname = filename[ib]
                parts = fname.rstrip("/").split("/")
                pt_folder = "/".join(parts[:-2]) + "/"
                slice_num = parts[-1].replace(".h5", ".bmp")
                pt_folder = pt_folder.replace("h5_data", f"generated_images/FMTA/{save_exp_name}")

                if not os.path.exists(pt_folder):
                    os.makedirs(pt_folder)

                fft1 = ft1[ib]
                fft2 = ft2[ib]
                fft3 = ft3[ib]
                save_img_name = f"{fft1}_{fft2}_{fft3}_{slice_num}"
                # print(save_img_name)
                save_img_path = os.path.join(pt_folder, save_img_name)
                # print(save_img_path)

                img = samples[ib].cpu().numpy()
                img = (img / 2.0 + 0.5) * 255.0
                img = img.clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                cv2.imwrite(save_img_path, img)
            # break


if __name__ == "__main__":

    main()
