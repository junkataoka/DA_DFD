import torch
import numpy as np
from torch.nn.init import xavier_uniform_
from helper import adjust_learning_rate
from losses import srcClassifyLoss, tarClassifyLoss, adversarialLoss

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



def train_avatar_batch(model, src_train_batch, tar_train_batch, 
                        src_train_dataloader, tar_train_dataloader, 
                        optimizer_dict, cur_epoch, logger, val_dict, args):

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

    # penalty parameter
    #lam = 2 / (1 + math.exp(-1 * 10 * cur_epoch / epochs)) - 1 
    loss_dict = {}
    p = float(cur_epoch) / 20
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    adjust_learning_rate(optimizer_dict["encoder"], args.lr, cur_epoch, args.epochs) # adjust learning rate
    model.train()
    optimizer_dict["encoder"].zero_grad()

    src_class_prob, src_domain_prob, _ = model(src_input)
    tar_class_prob, tar_domain_prob, _ = model(tar_input)

    loss_dict["src_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=src_domain_prob, 
                                                    index=src_idx, weights_ord=val_dict["src_weights_ord"], 
                                                    src=True, is_encoder=True)

    loss_dict["tar_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=tar_domain_prob, 
                                                    index=tar_idx, weights_ord=val_dict["tar_weights_ord"], 
                                                    src=False, is_encoder=True)

    loss_dict["src_loss_class"] = srcClassifyLoss(src_class_prob, src_target, 
                                                  index=src_idx, weights_ord=val_dict["src_weights_ord"])

    loss_dict["tar_loss_class"] = tarClassifyLoss(args=args, epoch=cur_epoch, tar_cls_p=tar_class_prob, 
                                                  target_ps_ord=val_dict["tar_label_ps_ord"], 
                                                  index=tar_idx, weights_ord=val_dict["tar_weights_ord"],
                                                  th=val_dict["th"])

    loss_dict["encoder_loss"]= alpha * (loss_dict["src_loss_domain"] + loss_dict["tar_loss_domain"]) + \
                                        loss_dict["src_loss_class"] + loss_dict["tar_loss_class"]
    
    loss_dict["encoder_loss"].backward()
    optimizer_dict["encoder"].step()
    logger.log(loss_dict)

    optimizer_dict["classifier"].zero_grad()
    src_class_prob, src_domain_prob, _ = model(src_input)
    tar_class_prob, tar_domain_prob, _ = model(tar_input)
    loss_dict["src_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=src_domain_prob, 
                                                    index=src_idx, weights_ord=val_dict["src_weights_ord"], 
                                                    src=True, is_encoder=False)

    loss_dict["tar_loss_domain"] = adversarialLoss(args=args, epoch=cur_epoch, prob_p_dis=tar_domain_prob, 
                                                    index=tar_idx, weights_ord=val_dict["tar_weights_ord"], 
                                                    src=False, is_encoder=True)

    loss_dict["src_loss_class"] = srcClassifyLoss(src_class_prob, src_target, 
                                                  index=src_idx, weights_ord=val_dict["src_weights_ord"])

    loss_dict["tar_loss_class"] = tarClassifyLoss(args=args, epoch=cur_epoch, tar_cls_p=tar_class_prob, 
                                                  target_ps_ord=val_dict["tar_label_ps_ord"], 
                                                  index=tar_idx, weights_ord=val_dict["tar_weights_ord"],
                                                  th=val_dict["th"])

    loss_dict["classifier_loss"]= alpha * (loss_dict["src_loss_domain"] + loss_dict["tar_loss_domain"]) + \
                                    loss_dict["src_loss_class"] + loss_dict["tar_loss_class"]

    loss_dict["classifier_loss"].backward()
    optimizer_dict["classifier"].step()
    loss_dict["epoch"] = cur_epoch
    logger.log(loss_dict)