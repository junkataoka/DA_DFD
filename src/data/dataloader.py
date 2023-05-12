from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import transformation

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, drop = False):
        x = np.load(file_path, allow_pickle = True)
        x=np.vstack(x).astype(np.float)
        a = x[:, 1:]
        if drop:
            a = np.delete(a, 0, 0)
        self.x_data = a
        
        para = x[:, 0]
        if drop:
            para = np.delete(para, 0, 0)
        para = np.transpose(para)
        num = np.array([0,1,2,3])
        num = num.shape[0]
        para = np.array(para, dtype = int)
        y = np.eye(num)[para]
        self.y_data = y
        

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx, :]) 
        
        y = torch.FloatTensor(self.y_data[idx, :])
        return x,y
