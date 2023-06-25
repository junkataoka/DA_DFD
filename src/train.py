import torch
import numpy as np
import torch.nn as nn
import argparse
from models.wdcnn import WDCNN1
from torch.nn.init import xavier_uniform_
import matplotlib.pylab as plt
import wandb
import os
from matplotlib.ticker import FuncFormatter
import math
from helper import adjust_learning_rate

hyperparameter_defaults = dict(
    epochs=70,
    batch_train=40,
    batch_val=50,
    batch_test=40,
    lr=0.0002,
    weight_decay=0.0005,
    r=0.02
)


def to_percent(temp, position):
    return '%1.0f' % (temp) + '%'

# model initialization 
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


def generate_dataset(data_dir, src_data, tar_data, src_domain, tar_domain):
    src_path = os.path.join(data_dir, src_data, f"{src_domain}.npz")
    tar_path = os.path.join(data_dir, tar_data, f"{tar_domain}.npz")
    src_x = np.load(src_path)['x']
    src_y = np.load(src_path)['y']
    tar_x = np.load(tar_path)['x']
    tar_y = np.load(tar_path)['y']
    src_dataset = torch.utils.data.TensorDataset(torch.from_numpy(src_x).float(), 
                                                 torch.from_numpy(src_y).long())   
    tar_dataset = torch.utils.data.TensorDataset(torch.from_numpy(tar_x).float(), 
                                                 torch.from_numpy(tar_y).long())
    return src_dataset, tar_dataset 

def train_wdcnn_batch(model, src_train_batch, tar_train_batch, 
                          src_train_dataloader, tar_train_dataloader, optimizer,
                          lr, cur_epoch, epochs, batch_size,
                          criterion, log_metrics):

    try:
        (src_input, src_target) = src_train_batch.__next__()[1]
    except StopIteration:
        src_train_batch = enumerate(src_train_dataloader)
        (src_input, src_target) = src_train_batch.__next__()[1]

    try:
        (tar_input, tar_target) = tar_train_batch.__next__()[1]
    except StopIteration:
        tar_train_batch = enumerate(tar_train_dataloader)
        (tar_input, tar_target) = tar_train_batch.__next__()[1]

    optimizer.zero_grad()
    src_input = src_input.unsqueeze(1).float().cuda()
    src_target = src_target.long().cuda()
    tar_input = tar_input.unsqueeze(1).float().cuda()
    tar_target = tar_target.long().cuda()
    s_domain_label = torch.zeros(batch_size).long().cuda()
    t_domain_label = torch.ones(batch_size).long().cuda()

    # penalty parameter
    #lam = 2 / (1 + math.exp(-1 * 10 * cur_epoch / epochs)) - 1 
    adjust_learning_rate(optimizer, lr, cur_epoch, epochs) # adjust learning rate

    model.train()
    p = float(cur_epoch) / 20
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    s_out_train, s_domain_out = model(src_input, alpha)
    _, t_domain_out = model(tar_input, alpha)
    loss_domain_s = criterion(s_domain_out, s_domain_label)
    loss_domain_t = criterion(t_domain_out, t_domain_label)

    loss_c = criterion(s_out_train, src_target)
    # loss = loss_c 
    loss = loss_c + (loss_domain_s + loss_domain_t)*0.5
    loss.backward()
    optimizer.step()

    log_metrics["loss_c"] = loss_c.item()
    log_metrics["loss_domain_s"] = loss_domain_s.item()
    log_metrics["loss_domain_t"] = loss_domain_t.item()
    log_metrics["loss"] = loss.item()

def train_avatar_batch(model, src_train_batch, tar_train_batch, 
                          src_train_dataloader, tar_train_dataloader, optimizer,
                          lr, cur_epoch, epochs, batch_size,
                          criterion, log_metrics):

    try:
        (src_input, src_target) = src_train_batch.__next__()[1]
    except StopIteration:
        src_train_batch = enumerate(src_train_dataloader)
        (src_input, src_target) = src_train_batch.__next__()[1]

    try:
        (tar_input, tar_target) = tar_train_batch.__next__()[1]
    except StopIteration:
        tar_train_batch = enumerate(tar_train_dataloader)
        (tar_input, tar_target) = tar_train_batch.__next__()[1]

    optimizer.zero_grad()
    src_input = src_input.unsqueeze(1).float().cuda()
    src_target = src_target.long().cuda()
    tar_input = tar_input.unsqueeze(1).float().cuda()
    tar_target = tar_target.long().cuda()
    s_domain_label = torch.zeros(batch_size).long().cuda()
    t_domain_label = torch.ones(batch_size).long().cuda()

    # penalty parameter
    #lam = 2 / (1 + math.exp(-1 * 10 * cur_epoch / epochs)) - 1 
    adjust_learning_rate(optimizer, lr, cur_epoch, epochs) # adjust learning rate

    model.train()
    p = float(cur_epoch) / 20
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    s_out_train, s_domain_out = model(src_input, alpha)
    _, t_domain_out = model(tar_input, alpha)
    loss_domain_s = criterion(s_domain_out, s_domain_label)
    loss_domain_t = criterion(t_domain_out, t_domain_label)

    loss_c = criterion(s_out_train, src_target)
    # loss = loss_c + (loss_domain_s + loss_domain_t)*0.5
    loss = loss_c 
    loss.backward()
    optimizer.step()

    log_metrics["loss_c"] = loss_c.item()
    log_metrics["loss_domain_s"] = loss_domain_s.item()
    log_metrics["loss_domain_t"] = loss_domain_t.item()
    log_metrics["loss"] = loss.item()
