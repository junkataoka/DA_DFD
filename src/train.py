import torch
import numpy as np
from torch.nn.init import xavier_uniform_
from helper import adjust_learning_rate, Augment
from losses import entropyMaxLoss
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
    
def adv_loss(prob_p_dis, src=True, is_encoder=True):

    if is_encoder:
        if src:
            loss_d = -(1-prob_p_dis).log().mean()
        else:
            loss_d = -prob_p_dis.log().mean()
    else:
        if src:
            loss_d = -prob_p_dis.log().mean()
        else:
            loss_d = -(1-prob_p_dis).log().mean()

    return loss_d

    



def train_batch(model, src_train_batch, tar_train_batch, 
                        src_train_dataloader, tar_train_dataloader, 
                        optimizers, criterion, cur_epoch, logger, args, backprop):

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
    p2 = float(cur_epoch-args.warmup_epoch) / args.epochs
    alpha = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1).cuda()
    alpha2 = torch.tensor((np.exp(10 * p2) - 1) / (np.exp(10) - 1)).cuda()
    adjust_learning_rate(optimizers["encoder"], args.lr, cur_epoch, args.epochs) # adjust learning rate
    adjust_learning_rate(optimizers["classifier"], args.lr, cur_epoch, args.epochs) # adjust learning rate
    model.train()

    # Encoder
    enc_src_dis_label = torch.ones(src_input.shape[0]).long().cuda()
    enc_tar_dis_label = torch.zeros(tar_input.shape[0]).long().cuda()
    #src_input_a, src_input_b = augmentation(src_input.unsqueeze(1))
    #_, _, src_feat_a = model(src_input_a.squeeze(1), alpha, True)
    #_, _, src_feat_b = model(src_input_b.squeeze(1), alpha, True)
    src_class, src_dis, _ = model(src_input, alpha, is_source=True)
    tar_class, tar_dis, _ = model(tar_input, alpha, is_source=False if args.use_domain_bn else True)

    enc_nll_src_class = src_class.log()
    enc_nll_src_dis = src_dis.log()
    enc_nll_tar_dis = tar_dis.log()
    enc_loss_tar_ent = entropyMaxLoss(tar_class, 0.9)

    enc_loss_src_cls = criterion(enc_nll_src_class, src_target.reshape(-1))
    #loss_src_contra = criterion_contra(src_feat_a, src_feat_b)
    enc_loss_src_dis = criterion(enc_nll_src_dis, enc_src_dis_label)
    enc_loss_tar_dis = criterion(enc_nll_tar_dis, enc_tar_dis_label)

    #loss = (loss_dis + loss_cls + loss_contra) / args.accum_iter
    loss = enc_loss_src_cls
    
    if args.use_domain_adv:
        loss += alpha*(enc_loss_src_dis + enc_loss_tar_dis)

    if args.use_tar_entropy and cur_epoch > args.warmup_epoch:
        loss += alpha*enc_loss_tar_ent

    loss.backward()
    optimizers["encoder"].step()
    optimizers["encoder"].zero_grad()

    # Classifier
    cls_src_dis_label = torch.zeros(src_input.shape[0]).long().cuda()
    cls_tar_dis_label = torch.ones(tar_input.shape[0]).long().cuda()
    #src_input_a, src_input_b = augmentation(src_input.unsqueeze(1))
    #_, _, src_feat_a = model(src_input_a.squeeze(1), alpha, True)
    #_, _, src_feat_b = model(src_input_b.squeeze(1), alpha, True)
    src_class, src_dis, _ = model(src_input, alpha, True)
    tar_class, tar_dis, _ = model(tar_input, alpha, is_source=False if args.use_domain_bn else True)

    cls_nll_src_class = src_class.log()
    cls_nll_src_dis = src_dis.log()
    cls_nll_tar_dis = tar_dis.log()
    cls_loss_tar_ent = entropyMaxLoss(tar_class, 0.9)

    cls_loss_src_cls = criterion(cls_nll_src_class, src_target.reshape(-1))
    #loss_src_contra = criterion_contra(src_feat_a, src_feat_b)
    cls_loss_src_dis = criterion(cls_nll_src_dis, cls_src_dis_label)
    cls_loss_tar_dis = criterion(cls_nll_tar_dis, cls_tar_dis_label)

    if args.pretrained:
        loss = cls_loss_src_cls
    else:
        loss = cls_loss_src_cls
    
    if args.use_domain_adv:
        loss += alpha*(cls_loss_src_dis + cls_loss_tar_dis)

    if args.use_tar_entropy and cur_epoch > args.warmup_epoch:
        loss += alpha*cls_loss_tar_ent

    loss.backward()
    optimizers["classifier"].step()
    optimizers["classifier"].zero_grad()

        
    logger.log({"loss_dis": (cls_loss_src_dis+cls_loss_tar_dis+enc_loss_tar_dis+enc_loss_src_dis)/4,
                "loss_src_cls": (cls_loss_src_cls+enc_loss_src_cls)/2,
                "loss_tar_ent": (cls_loss_tar_ent+enc_loss_tar_ent)/2,
                #"loss_contra": loss_contra,
                #"loss_src_contra": loss_src_contra,
                #"loss_tar_contra": loss_tar_contra,
                "loss": loss})