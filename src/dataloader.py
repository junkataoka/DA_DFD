from torch.utils.data import Dataset, TensorDataset
import torch.utils.data as Data
import numpy as np
import torch
import os

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, x, y, transform=None):
        assert all(x[0].size(0) == tensor.size(0) for tensor in x)
        assert all(y[0].size(0) == tensor.size(0) for tensor in y)
        self.transform = transform
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform:
            x = self.transform(x)

        y = self.y[index]

        return index, x, y

    def __len__(self):
        return self.x.size(0)

def generate_dataset(data_dir, src_data, tar_data, src_domain, tar_domain):
    src_path = os.path.join(data_dir, src_data, f"{src_domain}.npz")
    tar_path = os.path.join(data_dir, tar_data, f"{tar_domain}.npz")
    # Need to reshape in to [B, C, L]
    src_x = np.load(src_path)['x']
    src_y = np.load(src_path)['y']
    tar_x = np.load(tar_path)['x']
    tar_y = np.load(tar_path)['y']
    src_dataset = CustomTensorDataset(torch.from_numpy(src_x).float(), 
                                                 torch.from_numpy(src_y).long())   
    tar_dataset = CustomTensorDataset(torch.from_numpy(tar_x).float(), 
                                                 torch.from_numpy(tar_y).long())
    return src_dataset, tar_dataset 

def test_generate_dataset():
    src, tar = generate_dataset("/data/home/jkataok1/DA_DFD/data/processed", 
                     "CWRU", "CWRU", 0, 1)
    assert src.x.shape[0] == src.y.shape[0]
    assert tar.x.shape[0] == tar.y.shape[0]
    

    
