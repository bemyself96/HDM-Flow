import shutil
import os
import torch
import math


def build_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
    min_lr_ratio=0.1,  # 最低学习率为初始 lr 的 10%
):

    assert 0 <= min_lr_ratio <= 1

    def lr_lambda(step):

        if step >= total_steps:
            return min_lr_ratio

        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)

        progress = (step - warmup_steps) / (total_steps - warmup_steps)

        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def backup_file(backup_dir, file_path="./"):

    for file in os.listdir(file_path):
        if file.endswith(".py"):
            shutil.copy(os.path.join(file_path, file), os.path.join(backup_dir, file))
