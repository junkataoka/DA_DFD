import torch.nn.functional as F
import math
import torch

def adversarialLoss(args, epoch, prob_p_dis, index, weights_ord, src=True, is_encoder=True):

    if not args.pretrained:
        return torch.tensor([0]).float().cuda()

    weights = weights_ord[index]

    if is_encoder:
        if src:
            loss_d = - ((weights) * ((1-prob_p_dis).log())).mean()
        else:
            if epoch < args.warmup_epoch:
                weights.fill_(1)
            loss_d = - ((weights) * (prob_p_dis.log())).mean()

    else:
        if src:
            loss_d = - ((weights) * (prob_p_dis.log())).mean()
        else:
            if epoch < args.warmup_epoch:
                weights.fill_(1)
            loss_d = - ((weights) * ((1-prob_p_dis).log())).mean()


    return loss_d

def tarClassifyLoss(args, epoch, tar_cls_p, target_ps_ord, index, weights_ord, th):

    if not args.pretrained:
        return torch.tensor([0]).float().cuda()

    prob_q2 = tar_cls_p / tar_cls_p.sum(0, keepdim=True).pow(0.5)
    prob_q2 /= prob_q2.sum(1, keepdim=True)
    prob_q = prob_q2

    tar_weights = weights_ord[index.cuda()]
    target_ps = target_ps_ord[index.cuda()]

    pos_mask = torch.where(tar_weights >= th[target_ps], 1, 0)

    if epoch < args.warmup_epoch:
        tar_weights.fill_(1)
        pos_loss = torch.tensor([0]).float().cuda()
        neg_loss = torch.tensor([0]).float().cuda()

    else:
        if len(torch.unique(pos_mask)) == 2:
            pos_loss = - (tar_weights[pos_mask==1] * (prob_q[pos_mask==1] * tar_cls_p[pos_mask==1].log()).sum(1)).mean()
            neg_loss = - ((1-tar_weights[pos_mask==0]) * (prob_q[pos_mask==0] * (1-tar_cls_p[pos_mask==0]).log()).sum(1)).mean()

        else:
            pos_loss = - (tar_weights[pos_mask==1] * (prob_q[pos_mask==1] * tar_cls_p[pos_mask==1].log()).sum(1)).mean()
            neg_loss = torch.tensor([0]).float().cuda()
    
    assert math.isnan(pos_loss) == False
    assert math.isnan(neg_loss) == False

    return pos_loss + neg_loss

def srcClassifyLoss(src_cls_p, target, index, weights_ord):

    prob_q = torch.zeros(src_cls_p.size(), dtype=torch.float).to(device=src_cls_p.device)
    prob_q.scatter_(dim=1, index=target, src=torch.ones(src_cls_p.size(0), 1).to(device=src_cls_p.device))

    src_weights = weights_ord[index].to(device=src_cls_p.device)
    pos_loss = - (src_weights * (prob_q * src_cls_p.log()).sum(1)).mean()

    return pos_loss
