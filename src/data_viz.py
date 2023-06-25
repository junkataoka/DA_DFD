#%%
import numpy as np
# %%
def summary(data):
    xShape = data["x"].shape
    yShape = data["y"].shape
    print("Total sumple size: ", xShape[0])
    n_unique, counts = np.unique(data['y'], return_counts=True)
    for idx, i in zip(n_unique, counts):
        print("Class {}: {}".format(idx, i))
# %%
data = np.load("/data/home/jkataok1/DA_DFD/data/processed/CWRU/3.npz")
summary(data)

# %%
