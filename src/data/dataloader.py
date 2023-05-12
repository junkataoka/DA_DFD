import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler
import glob
import os
import numpy as np

class ReflowDataset(Dataset):
    def __init__(self, geom, heatmap, recipe):
        self.geom = sorted(glob.glob(os.path.join(geom, "*")))
        self.heatmap = sorted(glob.glob(os.path.join(heatmap, "*")))
        self.recipe = sorted(glob.glob(os.path.join(recipe, "*")))
        self.recipe_num = [int(c.split("/")[-1].split(".")[0]) for c in self.recipe]

    def __len__(self):
        return len(self.recipe_num)

    def __getitem__(self, recipe_idx):

        heatmap = torch.stack([torch.tensor(np.loadtxt(c, delimiter=" ")) for c in self.heatmap if recipe_idx == int(c.split("/")[-1].split("-")[0])], dim=0)
        geom = torch.stack([torch.tensor(np.loadtxt(c, delimiter=" ")) for c in self.geom if recipe_idx == int(c.split("/")[-1].split("-")[0])], dim=0)

        x = torch.DoubleTensor(torch.stack([geom, heatmap], dim=1))
        y = torch.cat([torch.DoubleTensor(np.loadtxt(c, delimiter=" ")) for c in self.recipe if recipe_idx == int(c.split("/")[-1].split(".")[0])], dim=0)

        return x, y

def generate_dataloader(geom_path, heatmap_path, recipe_path, batch_size, train=True):

    dataset = ReflowDataset(geom_path, heatmap_path, recipe_path)

    if train:
        if len(dataset) < batch_size:
            indices = np.random.randint(0, len(dataset), batch_size)
            sampler = SubsetRandomSampler(indices)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        else:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True)


    else:

        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    return dataloader
