import torch
import numpy as np
from torch.nn.init import xavier_uniform_
from helper import adjust_learning_rate, Augment
from losses import srcClassifyLoss, tarClassifyLoss, adversarialLoss
import math
from torch.nn import functional as F


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



def train_batch(model, src_train_batch, tar_train_batch, 
                        src_train_dataloader, tar_train_dataloader, 
                        optimizer, criterion, criterion_contra, cur_epoch, logger, args, backprop):

    try:
        (src_idx, src_input, src_target) = src_train_batch.__next__()[1]
    except StopIteration:
        src_train_batch = enumerate(src_train_dataloader)
        (src_idx, src_input, src_target) = src_train_batch.__next__()[1]

    try:
        (tar_idx, tar_input, tar_target) = tar_train_batch.__next__()[1]
    except StopIteration:
        tar_train_batch = enumerate(tar_train_dataloader)
        (tar_idx, tar_input, tar_target) = tar_train_batch.__next__()[1]

    src_input = src_input.float().cuda()
    src_target = src_target.long().cuda()
    tar_input = tar_input.float().cuda()
    tar_target = tar_target.long().cuda()

    augmentation = Augment(img_size=(src_input.shape[-2], src_input.shape[-1]))

    # penalty parameter
    p = float(cur_epoch) / args.epochs
    alpha = torch.tensor([2. / (1. + np.exp(-10 * p)) - 1]).cuda()
    adjust_learning_rate(optimizer, args.lr, cur_epoch, args.epochs) # adjust learning rate
    model.train()

    # Source domain
    src_dis_label = torch.ones(src_input.shape[0]).long().cuda()
    src_input_a, src_input_b = augmentation(src_input.unsqueeze(1))
    _, _, src_feat_a = model(src_input_a.squeeze(1), alpha, True)
    _, _, src_feat_b = model(src_input_b.squeeze(1), alpha, True)
    src_class, src_dis, _ = model(src_input, alpha, True)

    nll_src_class = F.log_softmax(src_class, dim=1)
    nll_src_dis = F.log_softmax(src_dis, dim=1)

    loss_src_cls = criterion(nll_src_class, src_target.reshape(-1))
    loss_src_contra = criterion_contra(src_feat_a, src_feat_b)
    loss_src_dis = criterion(nll_src_dis, src_dis_label)

    # Target domain
    tar_dis_label = torch.zeros(tar_input.shape[0]).long().cuda()
    tar_input_a, tar_input_b = augmentation(tar_input.unsqueeze(1))
    _, _, tar_feat_a = model(tar_input_a.squeeze(1), alpha, args.use_domain_bn is False)
    _, _, tar_feat_b = model(tar_input_b.squeeze(1), alpha, args.use_domain_bn is False)
    tar_class, tar_dis, _ = model(tar_input, alpha, args.use_domain_bn is False)

    nll_tar_class = F.log_softmax(tar_class, dim=1)
    nll_tar_dis = F.log_softmax(tar_dis, dim=1)

    loss_tar_cls = criterion(nll_tar_class, tar_target.reshape(-1))
    loss_tar_contra = criterion_contra(tar_feat_a, tar_feat_b)
    loss_tar_dis = criterion(nll_tar_dis, tar_dis_label)


    loss_dis = (loss_src_dis + loss_tar_dis) / 2

    loss_cls = loss_src_cls

    loss = loss_cls

    loss_contra = (loss_src_contra + loss_tar_contra) / 2

    #loss = (loss_dis + loss_cls + loss_contra) / args.accum_iter
    if args.use_domain_adv:
        loss += loss_dis

    if args.use_contra_learn:
        loss += loss_contra

    loss /= (args.accum_iter)

    loss.backward()

    if backprop:
        optimizer.step()
        optimizer.zero_grad()
        
        logger.log({"loss_dis": loss_dis,
                    "loss_src_dis": loss_src_dis,
                    "loss_tar_dis": loss_tar_dis,
                    "loss_cls": loss_cls,
                    "loss_src_cls": loss_src_cls,
                    "loss_tar_cls": loss_tar_cls,
                    "loss_contra": loss_contra,
                    "loss_src_contra": loss_src_contra,
                    "loss_tar_contra": loss_tar_contra,
                    "loss": loss})