import torch
from torch import nn
import math


class LoraLayer(nn.Module):
    def __init__(self, raw_layer, in_feature, out_feature, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.raw_layer = raw_layer

        self.lora_a = nn.Parameter(torch.empty(in_feature, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_feature))

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        raw_x = self.raw_layer(x)
        lora_x = x @ ((self.lora_a @ self.lora_b) * self.alpha / self.rank)
        return raw_x + lora_x


def inject_lora(model, name, layer, rank, alpha):
    # blocks.0.attn.qkv
    name_cols = name.split(".")  # blocks 0 attn qkv
    children = name_cols[:-1]  # blocks 0 attn
    cur_layer = model
    for child in children:
        cur_layer = getattr(cur_layer, child)

    lora_layer = LoraLayer(layer, layer.in_features, layer.out_features, rank, alpha)
    setattr(cur_layer, name_cols[-1], lora_layer)
