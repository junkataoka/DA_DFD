#%0
from helper import matfile_to_df, get_df_all, scale_signal
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import CubicSpline
from scipy.signal import spectrogram

def create_dataset_cwru(datapath_list, segment_length=2048, normalize=True):
    res = []
    for df in datapath_list:
        df_temp = matfile_to_df(df)
        res.append(df_temp)
    df = pd.concat(res, axis=0)
    df_all, mean, std = get_df_all(df, segment_length, normalize)
    return df_all, mean, std

def save_cwru(df, savepath, condition, segment_length, fs=1200, nperseg=128):
    data = df[df["condition"].isin(condition)]
    y = data["label"].to_numpy().reshape(-1, 1)
    x = data.loc[:, ~data.columns.isin(["filename", "label", "condition"])].to_numpy()
    res_x = []
    for i in range(len(x)):
        f, t, Sxx = spectrogram(np.array(x[i]), fs=fs, nperseg=nperseg)
        res_x.append(Sxx)
    res_x = np.stack(res_x)
    res_x_norm = (res_x - res_x.mean(axis=(0, 2), keepdims=True)) / (res_x.std(axis=(0, 2), keepdims=True)+1e-8)

    x = StandardScaler().fit_transform(x.reshape(-1, 1))
    x = x.reshape(-1, 1, segment_length)
    np.savez(savepath, x=x, y=y)
    np.savez(savepath.replace(".npz", "_spectrogram.npz"), x=res_x_norm, y=y)


def generate_spectrogram(df, fs=1200, nperseg=20):
    res_x = []
    for i in range(len(df)):
        x = df.loc[i, ~df.columns.isin(['label', "filename"])].to_list()
        f, t, Sxx = spectrogram(np.array(x), fs=fs, nperseg=nperseg)
        res_x.append(Sxx)
    res_x = np.stack(res_x)
    res_x_norm = (res_x - res_x.mean(axis=(0, 2), keepdims=True)) / (res_x.std(axis=(0, 2), keepdims=True)+1e-8)
    return res_x_norm

def prepare_cwru(segment_length, normalize, fs, nperseg):
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
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/0.npz", condition=[0], segment_length=segment_length, fs=fs, nperseg=nperseg)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/1.npz", condition=[1], segment_length=segment_length, fs=fs, nperseg=nperseg)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/2.npz", condition=[2],  segment_length=segment_length , fs=fs, nperseg=nperseg)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/3.npz", condition=[3],  segment_length=segment_length, fs=fs, nperseg=nperseg)
    save_cwru(df=df, savepath="/data/home/jkataok1/DA_DFD/data/processed/CWRU/all.npz", condition=[0, 1, 2, 3],  segment_length=segment_length, fs=fs, nperseg=nperseg)
    np.savez("/data/home/jkataok1/DA_DFD/data/processed/CWRU/mean_std.npz", x=np.array([mean, std]))

def prepare_ims(segment_length=20480, fs=20480, nperseg=600):
    save_path = "/data/home/jkataok1/DA_DFD/data/processed/IMS/0.npz"
    datapath_normal = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/normal/*")
    datapath_inner = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/inner/*")
    datapath_outer = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/outer/*")
    datapath_ball = glob.glob("/data/home/jkataok1/DA_DFD/data/raw/IMS/ball/*")

    normal = [format_data_ims(datapath_normal[i], "normal", segment_length) for i in range(len(datapath_normal))]
    normal_X = np.concatenate([normal[i][0] for i in range(len(normal))] , axis=0)
    normal_y = np.concatenate([normal[i][1] for i in range(len(normal))] , axis=0)
    
    inner = [format_data_ims(datapath_inner[i], "inner", segment_length) for i in range(len(datapath_inner))]
    inner_X = np.concatenate([inner[i][0] for i in range(len(inner))] , axis=0)
    inner_y = np.concatenate([inner[i][1] for i in range(len(inner))] , axis=0)

    
    outer = [format_data_ims(datapath_outer[i], "outer", segment_length) for i in range(len(datapath_outer))]
    outer_X = np.concatenate([outer[i][0] for i in range(len(outer))] , axis=0)
    outer_y = np.concatenate([outer[i][1] for i in range(len(outer))] , axis=0)

    ball = [format_data_ims(datapath_ball[i], "ball", segment_length) for i in range(len(datapath_ball))]
    ball_X = np.concatenate([ball[i][0] for i in range(len(ball))] , axis=0)
    ball_y = np.concatenate([ball[i][1] for i in range(len(ball))] , axis=0)
    mean_std = np.load("/data/home/jkataok1/DA_DFD/data/processed/CWRU/mean_std.npz")["x"]

    X = np.concatenate([normal_X, inner_X, outer_X, ball_X], axis=0)
    res_x = []
    for i in range(len(X)):
        f, t, Sxx = spectrogram(np.array(X[i]), fs=fs, nperseg=nperseg)
        res_x.append(Sxx)
    res_x = np.stack(res_x)
    res_x_norm = (res_x - res_x.mean(axis=(0, 2), keepdims=True)) / (res_x.std(axis=(0, 2), keepdims=True)+1e-8)
    # X = scale_signal(X, mean_std[0], mean_std[1])
    X = StandardScaler().fit_transform(X.reshape(-1, 1))
    #X = StandardScaler().fit_transform(X)
    X = X.reshape(-1, 1, segment_length)
    y = np.concatenate([normal_y, inner_y, outer_y, ball_y], axis=0)
    y = y.reshape(-1, 1)
    np.savez(save_path, x=X, y=y)
    np.savez(save_path.replace(".npz", "_spectrogram.npz"), x=res_x_norm, y=y)

    
   
def format_data_ims(file_name, fault_pattern, segment_length=2048):

    fault_column = {"normal": 0, "inner": 4, "ball": 6, "outer": 2}
    label = {"normal": 0, "ball": 1, "inner": 2, "outer": 3}
    temp = pd.read_csv(open(file_name,'r'), delim_whitespace=True, header=None)
    N = temp.shape[0]
    val = temp[fault_column[fault_pattern]].values
    splitted_val = np.stack(np.array_split(val, N//segment_length))

    return splitted_val, np.repeat(label[fault_pattern], N//segment_length)
    
def main():
    prepare_cwru(2048, False, 2048, 128)
    prepare_ims(2048, 2048, 128)


# %%
if __name__ == "__main__":
    main()

