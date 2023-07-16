import pytest
from src.dataloader import generate_dataset

def generate_dataset():
    src, tar = generate_dataset("/data/home/jkataok1/DA_DFD/data/processed", 
                     "CWRU", "CWRU", 0, 1)
    assert src.x.shape[0] == src.y.shape[0]
    assert tar.x.shape[0] == tar.y.shape[0]
    

    
