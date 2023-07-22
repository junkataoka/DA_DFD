#%0
from helper import matfile_to_df, get_df_all, scale_signal
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import CubicSpline

def create_dataset_cwru(datapath_list, segment_length=2048, normalize=True):
    res = []
    for df in datapath_list:
        df_temp = matfile_to_df(df)
        res.append(df_temp)
    df = pd.concat(res, axis=0)
    df_all, mean, std = get_df_all(df, segment_length, normalize)
    return df_all, mean, std

def save_cwru(df, savepath, condition, segment_length):
    data = df[df["condition"].isin(condition)]
    y = data["label"].to_numpy().reshape(-1, 1)
    x = data.loc[:, ~data.columns.isin(["filename", "label", "condition"])].to_numpy()
    x = StandardScaler().fit_transform(x.reshape(-1, 1))
    #x = StandardScaler().fit_transform(x)
    x = x.reshape(-1, 1, segment_length)
    np.savez(savepath, x=x, y=y)

def prepare_cwru(segment_length, normalize):

    datapath_de12 = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/12k_DE"
    datapath_nor = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/Normal"
    datapath_de48 = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/48k_DE"
    datapath_fe12 = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/12k_FE"
    datapath_list = [datapath_de12, datapath_nor, datapath_fe12]
    df, mean, std = create_dataset_cwru(datapath_list, segment_length, normalize)
    data_path = df.iloc[:, 1]
    temp = data_path.tolist()
    condition = [int(temp[i].split('/')[-1].split('_')[-1].split(".")[0]) for i in range(len(temp))]
    df["condition"] = condition
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/0.npz", condition=[0], segment_length=segment_length)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/1.npz", condition=[1], segment_length=segment_length)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/2.npz", condition=[2],  segment_length=segment_length)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/3.npz", condition=[3],  segment_length=segment_length)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/all.npz", condition=[0, 1, 2, 3],  segment_length=segment_length)
    np.savez("/data/home/jkataok1/DA_DFD/data/processed/CWRU/mean_std.npz", x=np.array([mean, std]))

def prepare_ims(segment_length=2048):
    save_path = "/data/home/jkataok1/DA_DFD/data/processed/IMS/0.npz"
    datapath_normal = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/normal/*")
    datapath_inner = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/inner/*")
    datapath_outer = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/outer/*")
    datapath_ball = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/ball/*")

    normal = [format_data_ims(datapath_normal[i], "normal") for i in range(len(datapath_normal))]
    normal_X = np.concatenate([normal[i][0] for i in range(len(normal))] , axis=0)
    normal_y = np.concatenate([normal[i][1] for i in range(len(normal))] , axis=0)
    
    inner = [format_data_ims(datapath_inner[i], "inner") for i in range(len(datapath_inner))]
    inner_X = np.concatenate([inner[i][0] for i in range(len(inner))] , axis=0)
    inner_y = np.concatenate([inner[i][1] for i in range(len(inner))] , axis=0)

    
    outer = [format_data_ims(datapath_outer[i], "outer") for i in range(len(datapath_outer))]
    outer_X = np.concatenate([outer[i][0] for i in range(len(outer))] , axis=0)
    outer_y = np.concatenate([outer[i][1] for i in range(len(outer))] , axis=0)

    ball = [format_data_ims(datapath_ball[i], "ball") for i in range(len(datapath_ball))]
    ball_X = np.concatenate([ball[i][0] for i in range(len(ball))] , axis=0)
    ball_y = np.concatenate([ball[i][1] for i in range(len(ball))] , axis=0)
    mean_std = np.load("/data/home/jkataok1/DA_DFD/data/processed/CWRU/mean_std.npz")["x"]

    X = np.concatenate([normal_X, inner_X, outer_X, ball_X], axis=0)
    # X = scale_signal(X, mean_std[0], mean_std[1])
    X = StandardScaler().fit_transform(X.reshape(-1, 1))
    #X = StandardScaler().fit_transform(X)
    X = X.reshape(-1, 1, segment_length)
    y = np.concatenate([normal_y, inner_y, outer_y, ball_y], axis=0)
    y = y.reshape(-1, 1)
    np.savez(save_path, x=X, y=y)

    
   
def format_data_ims(file_name, fault_pattern, segment_length=2048):

    fault_column = {"normal": 0, "inner": 4, "ball": 6, "outer": 2}
    label = {"normal": 0, "ball": 1, "inner": 2, "outer": 3}
    temp = pd.read_csv(open(file_name,'r'), delim_whitespace=True, header=None)
    N = temp.shape[0]
    val = temp[fault_column[fault_pattern]].values
    #val = StandardScaler().fit_transform(val.reshape(-1, 1))
    #val = val.reshape(1, -)
    #cs = CubicSpline(np.arange(N), val)
    #xs = np.arange(0, N, 0.1)
    #val = cs(xs)
    splitted_val = np.stack(np.array_split(val, N//segment_length))

    return splitted_val, np.repeat(label[fault_pattern], N//segment_length)
    
def main():
    prepare_cwru(2048, False)
    prepare_ims(2048)


# %%
if __name__ == "__main__":
    main()

