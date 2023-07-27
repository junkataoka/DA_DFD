import torchmetrics
import torch
from avatar import WAVATAR
from dataloader import generate_dataset
from helper import sort_list


def validate(model, src_dataloader, tar_dataloader, num_classes):
    model.eval()
    s_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    t_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    out_dict = {}
    with torch.no_grad():
        for i, (idx, src_input, src_target) in enumerate(src_dataloader):

            src_input = src_input.float().cuda()
            src_target = src_target.long().cuda()
            src_cls = model(src_input)
            src_cls_p = torch.softmax(src_cls, dim=-1)

            src_onehot = torch.cuda.FloatTensor(src_cls_p.size()).fill_(0)
            src_onehot.scatter_(1, src_target, torch.ones(src_cls_p.size(0), 1).cuda())
            s_cls_metric(src_cls_p.argmax(-1).cpu(), src_target.reshape(-1).cpu())

        s_acc_all = s_cls_metric.compute()  
        out_dict["src_acc"] = s_acc_all

    with torch.no_grad():
        for i, (idx, tar_input, tar_target) in enumerate(tar_dataloader):

            tar_input = tar_input.float().cuda()
            tar_target = tar_target.long().cuda()
            tar_cls = model(tar_input)
            tar_cls_p = torch.softmax(tar_cls, dim=-1)

            tar_onehot = torch.cuda.FloatTensor(tar_cls_p.size()).fill_(0)
            tar_onehot.scatter_(1, tar_target, torch.ones(tar_cls_p.size(0), 1).cuda())
            t_cls_metric(tar_cls_p.argmax(-1).cpu(), tar_target.reshape(-1).cpu())

        s_acc_all = s_cls_metric.compute()  
        out_dict["tar_acc"] = t_cls_metric.compute()  

    return out_dict

