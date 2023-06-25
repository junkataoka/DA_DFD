#%0
from helper import matfile_to_df, get_df_all
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

datapath_de12 = "/data/home/jkataok1/DA_DFD/data/raw/12k_DE"
datapath_nor = "/data/home/jkataok1/DA_DFD/data/raw/Normal"
datapath_de48 = "/data/home/jkataok1/DA_DFD/data/raw/48k_DE"
datapath_fe12 = "/data/home/jkataok1/DA_DFD/data/raw/12k_FE"
datapath_list = [datapath_de12, datapath_nor, datapath_fe12]

def create_dataset(datapath_list, segment_length=2048, normalize=True):
    res = []
    for df in datapath_list:
        df_temp = matfile_to_df(df)
        res.append(df_temp)
    df = pd.concat(res, axis=0)
    df_all = get_df_all(df, segment_length, normalize)
    return df_all

def save_cwrs(df, savepath, condition):
    data = df[df["condition"]==condition]
    y = data["label"].to_numpy()
    x = data.loc[:, ~data.columns.isin(["filename", "label", "condition"])].to_numpy()
    np.savez(savepath, x=x, y=y)

def prepare_cwrs(datapath_list, segment_length, normalize, test_ratio=0.25):
    df = create_dataset(datapath_list, segment_length, normalize)
    data_path = df.iloc[:, 1]
    temp = data_path.tolist()
    condition = [int(temp[i].split('/')[-1].split('_')[-1].split(".")[0]) for i in range(len(temp))]
    df["condition"] = condition
    save_cwrs(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/0.npz", condition=0)
    save_cwrs(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/1.npz", condition=1)
    save_cwrs(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/2.npz", condition=2)
    save_cwrs(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/3.npz", condition=3)

    
def main():
    prepare_cwrs(datapath_list, 2048, True)

# %%
if __name__ == "__main__":
    main()

