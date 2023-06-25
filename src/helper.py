# Helper functions to read and preprocess data files from Matlab format
# Data science libraries
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
# Others
from pathlib import Path
from tqdm.auto import tqdm
import requests
import os
from glob import glob
from torch.nn.init import xavier_uniform_
import math

def matfile_to_dic(folder_path):
    '''
    Read all the matlab files of the CWRU Bearing Dataset and return a 
    dictionary. The key of each item is the filename and the value is the data 
    of one matlab file, which also has key value pairs.
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic: 
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {}
    for _, filepath in enumerate(glob(os.path.join(folder_path, '*.mat'))):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath)
    return output_dic


def remove_dic_items(dic):
    '''
    Remove redundant data in the dictionary returned by matfile_to_dic inplace.
    '''
    # For each file in the dictionary, delete the redundant key-value pairs
    for _, values in dic.items():
        del values['__header__']
        del values['__version__']    
        del values['__globals__']


def rename_keys(dic):
    '''
    Rename some keys so that they can be loaded into a 
    DataFrame with consistent column names
    '''
    # For each file in the dictionary
    for _,v1 in dic.items():
        # For each key-value pair, rename the following keys 
        for k2,_ in list(v1.items()):
            if 'DE_time' in k2:
                v1['DE_time'] = v1.pop(k2)
            elif 'BA_time' in k2:
                v1['BA_time'] = v1.pop(k2)
            elif 'FE_time' in k2:
                v1['FE_time'] = v1.pop(k2)
            elif 'RPM' in k2:
                v1['RPM'] = v1.pop(k2)


def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    if 'B' in filename:
        return 'B'
    elif 'IR' in filename:
        return 'IR'
    elif 'OR' in filename:
        return 'OR'
    elif 'Normal' in filename:
        return 'N'


def matfile_to_df(folder_path):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        DataFrame with preprocessed data
    '''
    dic = matfile_to_dic(folder_path)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df.drop(['BA_time','FE_time', 'RPM', 'ans'], axis=1, errors='ignore')

def divide_signal(df, segment_length):

    dic = {}
    idx = 0
    for i in range(df.shape[0]): # 파일 개수
        n_sample_points = len(df.iloc[i,1]) # 파일 안에 있는 신호 개수 # 예를들어 122571
        n_segments = n_sample_points // segment_length # 원하는 segment 길이만큼 슬라이싱 할 개수 # 예를 들어 segment_length는 500, n_segments는 245
        for segment in range(n_segments): 
            dic[idx] = { # 
                'signal': df.iloc[i,1][segment_length * segment:segment_length * (segment+1)], 
                'label': df.iloc[i,2],
                'filename' : df.iloc[i,0]
            }
            idx += 1
    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    df_output = pd.concat(
        [df_tmp[['label', 'filename']], 
         pd.DataFrame(np.hstack(df_tmp["signal"].values).T)
        ], 
        axis=1 )
    return df_output

def normalize_signal(df):
    mean = df['DE_time'].apply(np.mean)
    std = df['DE_time'].apply(np.std)
    df['DE_time'] = (df['DE_time'] - mean) / std

def normalize_one_signal(df):
    mean = np.mean(df)
    std = np.std(df)
    df = (df - mean) / std
    return df

def get_df_all(df, segment_length=512, normalize=False):

    if normalize:
        normalize_signal(df)
    df_processed = divide_signal(df, segment_length)

    map_label = {'N':0, 'B':1, 'IR':2, 'OR':3}
    df_processed['label'] = df_processed['label'].map(map_label)
    return df_processed

def count_epoch_on_large_dataset(train_loader_target, train_loader_source):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    batch_number_s = len(train_loader_source)
    if batch_number_s > batch_number_t:
        batch_number = batch_number_s
    return batch_number

def weight_init(m):
    class_name = m.__class__.__name__ 
    if class_name.find('Conv') != -1:
        xavier_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        xavier_uniform_(m.weight.data)

# batch norm initialization
def batch_norm_init(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.reset_running_stats()

def adjust_learning_rate(optimizer, lr, cur_epoch, epochs):
    """Adjust the learning rate according the epoch"""
    lr = lr / math.pow((1 + 10 * float(cur_epoch) / float(epochs)), 0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

def obrain_params(model):
    params = list()

    for k, v in model.named_parameters():
        if "fc" in k:
            params += [{'params': v, 'name': "fc"}]
        elif "classifier" in k:
            params += [{'params': v, 'name': "classifier"}]
        else:
            params += [{'params': v, 'name': "feature"}]
    return params