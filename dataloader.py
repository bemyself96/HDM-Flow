import os
import h5py
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]
    return data


def process_image(image_path):
    image = load_h5(image_path)
    image = torch.from_numpy(image / 255.0)
    image = (image - 0.5) * 2.0
    return image


def vae_image(image_path):
    image_path = image_path.replace("h5_data", "vae_data")
    image_path = image_path.replace(".h5", ".npy")
    image = np.load(image_path)
    image = torch.from_numpy(image)
    return image


class Data_Loader(Dataset):
    def __init__(self, sdata_path, img_list, cdata_path):

        self.img_list = img_list
        self.sdata_path = sdata_path
        self.cdata_path = cdata_path

        self.path_cache = {}
        for i, sample in enumerate(self.img_list):
            self.path_cache[i] = self.sdata_path if "W" in sample[0] else self.cdata_path

    def __getitem__(self, index):

        sample_name_json = self.img_list[index]
        data_path = self.path_cache[index]

        ft1 = sample_name_json[0].split("/")[-2]  # W0
        ft2 = sample_name_json[1].split("/")[-2]  # W4
        ft3 = sample_name_json[2].split("/")[-2]  # W8

        image_t1_path = os.path.join(data_path, sample_name_json[0])
        image_t2_path = os.path.join(data_path, sample_name_json[1])
        image_t3_path = os.path.join(data_path, sample_name_json[2])

        t12 = float(sample_name_json[3])
        t23 = float(sample_name_json[4])

        alpha = float(t23 / t12)

        vae_t1 = vae_image(image_t1_path)
        vae_t2 = vae_image(image_t2_path)
        vae_t3 = vae_image(image_t3_path)

        sample = {
            "vae_t1": vae_t1.float(),
            "vae_t2": vae_t2.float(),
            "vae_t3": vae_t3.float(),
            "t23": t23,
            "alpha": alpha,
            "name": image_t3_path,
            "ft1": ft1,
            "ft2": ft2,
            "ft3": ft3,
        }

        return sample

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":

    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = Data_Loader(img_list=[])
    print("数据个数：", len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for batch, sample in enumerate(train_loader):
        print(sample["t12"].shape)

    # pass
