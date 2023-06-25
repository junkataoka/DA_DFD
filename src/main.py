# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import click
import seaborn as sns
import torch.utils.data as Data
from helper import count_epoch_on_large_dataset, weight_init, batch_norm_init, obrain_params
from models.wdcnn import WDCNN1
from validation import validate
from train import train_wdcnn_batch, generate_dataset
import torch
import os
import wandb
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='DA_DFD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=48, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
parser.add_argument('--momentum', type=float, default=1e-2, help='weight decay')
parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
parser.add_argument('--device', type=str, default="cuda:0", help='cuda device')
parser.add_argument('--data_path', type=str, default="/data/home/jkataok1/DA_DFD/data/processed", help='data path')
parser.add_argument('--src_data', type=str, default="CWRU", help='source data')
parser.add_argument('--tar_data', type=str, default="CWRU", help='target data')
parser.add_argument('--src_domain', type=str, default="0", help='source domain')
parser.add_argument('--tar_domain', type=str, default="1", help='target domain')
parser.add_argument('--log', type=str, default="log", help='log')
parser.add_argument('--pretrained_path', type=str, default="/data/home/jkataok1/DA_DFD/log/CWRS0toCWRS0_lr0.001_e2_b128/src_model.pth", help='pretrained_model_path')
parser.add_argument('--pretrained', action='store_true')



args = parser.parse_args()


def main(args):

    # create log directory
    log = args.log + "/" + f"{args.src_data}" + f"{args.src_domain}" + "to" + f"{args.tar_data}" + f"{args.tar_domain}" \
                + "_lr" + f"{str(args.lr)}" + "_e" + f"{str(args.epochs)}" + "_b" + f"{str(args.batch_size)}"

    # initialize metric
    log_metrics = {}

    # create log directory
    if not os.path.isdir(log):
        os.makedirs(log)
        print("create new log directory")

    # initialize wandb
    hyperparameter_defaults = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        src_domain=args.src_domain,
        tar_domain=args.tar_domain
    )
    wandb.init(config=hyperparameter_defaults, project="DA_DFD")
    wandb.define_metric("src_acc", summary="max")
    wandb.define_metric("tar_acc", summary="max")


    # generate dataset for training
    src_dataset, tar_dataset = generate_dataset(args.data_path, args.src_data, args.tar_data, args.src_domain, args.tar_domain)
    # convert dataset to dataloader
    src_train_dataloader = Data.DataLoader(src_dataset, 
                                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    tar_train_dataloader = Data.DataLoader(tar_dataset,
                                        batch_size=args.batch_size, shuffle=True, drop_last=True) 

    src_val_dataloader = Data.DataLoader(src_dataset, 
                                        batch_size=args.batch_size, shuffle=True, drop_last=False)
    tar_val_dataloader = Data.DataLoader(tar_dataset,
                                        batch_size=args.batch_size, shuffle=True, drop_last=False) 
    
    # define model 
    model = WDCNN1(C_in=1, class_num=args.num_classes).to(args.device)
    model.apply(weight_init)
    model.apply(batch_norm_init)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained_path))
        print("load pretrained model")

    # define optimizer 
    params = obrain_params(model)

    optimizer = torch.optim.SGD(params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    batch_count = count_epoch_on_large_dataset(src_train_dataloader, tar_train_dataloader)
    num_itern_total = args.epochs * batch_count
    epoch = 1
    best_acc = 0.0
    criterion = torch.nn.NLLLoss()
    count_itern_each_epoch = 0

    for itern in range(epoch*batch_count, num_itern_total):
        train_loader_source_batch = enumerate(src_train_dataloader)
        train_loader_target_batch = enumerate(tar_train_dataloader)

        if (itern==0 or count_itern_each_epoch==batch_count):
            wandb.log({"epoch": epoch})
            s_acc, t_acc = validate(model, src_val_dataloader, tar_val_dataloader, args.num_classes, wandb)
            if itern != 0:
                count_itern_each_epoch = 0
                epoch += 1

            if t_acc > best_acc:
                best_acc = t_acc
                torch.save(model.state_dict(), os.path.join(log, 'best_model.pth'))


        train_wdcnn_batch(model, train_loader_source_batch, train_loader_target_batch, 
                          src_train_dataloader, tar_train_dataloader, optimizer, 
                          args.lr, epoch, args.epochs, args.batch_size, criterion, log_metrics)
        wandb.log(log_metrics)
        count_itern_each_epoch += 1
    

if __name__ == "__main__":
    main(args)
