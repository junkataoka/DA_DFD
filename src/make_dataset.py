#%0
from helper import matfile_to_df, get_df_all, scale_signal
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.signal import spectrogram
import os

def create_dataset_cwru(datapath_list, segment_length=2048, normalize=True):
    res = []
    for df in datapath_list:
        df_temp = matfile_to_df(df)
        res.append(df_temp)
    df = pd.concat(res, axis=0)
    df_all, mean, std = get_df_all(df, segment_length, normalize)
    return df_all, mean, std

def save_cwru(df, savepath, condition, segment_length, fs=1200, nperseg=128, diameter="014"):
    data = df[df["condition"].isin(condition)]
    if diameter != "all":
        data = data[data["filename"].str.contains(diameter)]

    y = data["label"].to_numpy().reshape(-1, 1)
    x = data.loc[:, ~data.columns.isin(["filename", "label", "condition"])].to_numpy()
    res_x = []
    for i in range(len(x)):
        f, t, Sxx = spectrogram(np.array(x[i]), fs=fs, nperseg=nperseg)
        res_x.append(Sxx)
    res_x = np.stack(res_x)
    res_x_norm = (res_x - res_x.mean(axis=(0, -2, -1), keepdims=True)) / (res_x.std(axis=(0, -2, -1), keepdims=True)+1e-8)
    res_x_norm = np.expand_dims(res_x_norm, axis=1)

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
    res_x_norm = (res_x - res_x.mean(axis=(0, -2, -1), keepdims=True)) / (res_x.std(axis=(0, -2, -1), keepdims=True)+1e-8)
    return res_x_norm

def prepare_cwru(segment_length, normalize, fs, nperseg, diameter):
    datapath_de12 = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/12k_DE"
    datapath_nor = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/Normal"
    datapath_de48 = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/48k_DE"
    datapath_fe12 = "/data/home/jkataok1/DA_DFD/data/raw/CWRU/12k_FE"
    #datapath_list = [datapath_de12, datapath_nor, datapath_fe12, datapath_de48]
    datapath_list = [datapath_de12, datapath_nor, datapath_fe12]
    #datapath_list = [c for c in datapath_list if diameter in c]
    df, mean, std = create_dataset_cwru(datapath_list, segment_length, normalize)
    data_path = df.iloc[:, 1]
    temp = data_path.tolist()
    condition = [int(temp[i].split('/')[-1].split('_')[-1].split(".")[0]) for i in range(len(temp))]
    df["condition"] = condition
    save_cwru(df=df, savepath=f"/data/home/jkataok1/DA_DFD/data/processed/CWRU/0_{diameter}.npz", condition=[0], segment_length=segment_length, fs=fs, nperseg=nperseg, diameter=diameter)
    save_cwru(df=df, savepath=f"/data/home/jkataok1/DA_DFD/data/processed/CWRU/1_{diameter}.npz", condition=[1], segment_length=segment_length, fs=fs, nperseg=nperseg, diameter=diameter)
    save_cwru(df=df, savepath=f"/data/home/jkataok1/DA_DFD/data/processed/CWRU/2_{diameter}.npz", condition=[2],  segment_length=segment_length , fs=fs, nperseg=nperseg, diameter=diameter)
    save_cwru(df=df, savepath=f"/data/home/jkataok1/DA_DFD/data/processed/CWRU/3_{diameter}.npz", condition=[3],  segment_length=segment_length, fs=fs, nperseg=nperseg, diameter=diameter)
    save_cwru(df=df, savepath=f"/data/home/jkataok1/DA_DFD/data/processed/CWRU/all_{diameter}.npz", condition=[0, 1, 2, 3],  segment_length=segment_length, fs=fs, nperseg=nperseg, diameter=diameter)
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

def prepare_gearbox(datapath, segment_length=2048, nperseg=256):
    dataset_files = glob.glob(os.path.join(datapath, "*.csv"))



def format_gearbox(file_name, segment_length, nperseg):
    X = []
    y = []

    if "20_0" in file_name:
        domain = 0 
        frequency = 20
        sep = ","

    elif "30_2" in file_name:
        domain = 1
        frequency = 30
        sep = "\t"
    
    df = pd.read_csv(file_name, skiprows=16, header=None, usecols=range(8), sep=sep)
    cols = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]
    df.columns = cols

    if "ball" in file_name:
        df["label"] = 0

    elif "comb" in file_name:
        df["label"] = 1

    elif "inner" in file_name:
        df["label"] = 2

    elif "outer" in file_name:
        df["label"] = 3

    elif "health" in file_name:
        df["label"] = 4

    N = df.shape[0]
    val = df[cols].values
    splitted_val = np.stack(np.array_split(val[:int(N//segment_length * segment_length)], N//segment_length))
    for i in range(splitted_val.shape[0]):
        for j in range(splitted_val.shape[2]):
            f, t, Sxx = spectrogram(splitted_val[i, :, j], fs=frequency, nperseg=nperseg)
            X.append(Sxx)
            y.append(df['label'][0])

    X = np.stack(X)
    y = np.stack(y)

    return X, y
   
def format_data_ims(file_name, fault_pattern, segment_length=2048):

    fault_column = {"normal": 0, "inner": 4, "ball": 6, "outer": 2}
    label = {"normal": 0, "ball": 1, "inner": 2, "outer": 3}
    temp = pd.read_csv(open(file_name,'r'), delim_whitespace=True, header=None)
    N = temp.shape[0]
    val = temp[fault_column[fault_pattern]].values
    splitted_val = np.stack(np.array_split(val, N//segment_length))

    return splitted_val, np.repeat(label[fault_pattern], N//segment_length)

def prepare_PU():

    signal_size = 2048
    root = "/data/home/jkataok1/DA_DFD/data/raw/PU"
    #bearing_codes = ["K001", "KI14", "KI18", "KA04", "KA16", "KB23", "KB24"]
    working_condtions = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]

    healthy_code = ["K001", "K002", "K003", "K004", "K005"]
    outer_rind_code = ["KA04", "KA15", "KA16", "KA22", "KA30"]
    inner_ring_code = ["KI01", "KI14", "KI16", "KI18", "KI21"]

    bearing_codes = healthy_code + outer_rind_code + inner_ring_code

    def data_load(filename,name):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        '''
        fl = loadmat(filename)[name]
        fl = fl[0][0][2][0][6][2]  #Take out the data
        fl = fl.reshape(-1,1)
        fl = (fl - fl.mean(axis=0)) / fl.std(axis=0)
        arr_spectrogram = [] 
        arr_signal = []
        start,end = 0, signal_size
        while end<=fl.shape[0]:
            #data.append(fl[start:end])
            signal = fl[start:end].reshape(-1)
            f, t, sxx = spectrogram(signal, fs=2048, nperseg=128)
            arr_spectrogram.append(sxx)
            arr_signal.append(signal)
            start += signal_size
            end += signal_size

        data_spectrogram = np.stack(arr_spectrogram, axis=0)
        data_signal = np.stack(arr_signal)


        return data_spectrogram, data_signal


    def load_pu(root, bearing_codes, working_condtion):
        data_spectrogram = []
        data_signal = []
        label = []
        for i in range(len(bearing_codes)):
            file_names = glob.glob(os.path.join(root,  bearing_codes[i], working_condtion + '*'))
            for f in file_names:
                f_name = os.path.basename(f).split('.')[0]

                f_data_spectrogram, f_data_signal = data_load(f, f_name)
                data_spectrogram.append(f_data_spectrogram)
                data_signal.append(f_data_signal)
                if bearing_codes[i] in healthy_code:
                    label.append(np.repeat(0, len(f_data_spectrogram)))
                elif bearing_codes[i] in outer_rind_code:
                    label.append(np.repeat(1, len(f_data_spectrogram)))
                elif bearing_codes[i] in inner_ring_code:
                    label.append(np.repeat(2, len(f_data_spectrogram)))

        data_signal = np.expand_dims(np.concatenate(data_signal, axis=0), axis=1)
        data_spectrogram = np.expand_dims(np.concatenate(data_spectrogram), axis=1)
        data_spectrogram = (data_spectrogram - data_spectrogram.mean(axis=(0, -2, -1), keepdims=True)) / (data_spectrogram.std(axis=(0, -2, -1), keepdims=True)+1e-8)
        label = np.expand_dims(np.concatenate(label), axis=-1)

        return data_spectrogram, data_signal, label

    save_path = "/data/home/jkataok1/DA_DFD/data/processed/PU"
    for i in range(len(working_condtions)):
        data_spectrgram, data_signal, label = load_pu(root, bearing_codes, working_condtions[i])
        np.savez(os.path.join(save_path, f'{i}_spectrogram.npz'), x=data_spectrgram, y=label)
        np.savez(os.path.join(save_path, f'{i}.npz'), x=data_signal, y=label)

    
def main():
    prepare_PU()
    #prepare_cwru(2048, False, 2048, 128, "all")
    #prepare_ims(2048, 2048, 128)


# %%
if __name__ == "__main__":
    main()

