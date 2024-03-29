import torchmetrics
import torch
from avatar import WAVATAR
from dataloader import generate_dataset
from helper import sort_list
from torch.nn import functional as F


def validate(model, src_dataloader, tar_dataloader, args):
    model.eval()
    s_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
    t_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
    out_dict = {}

    with torch.no_grad():
        for i, (idx, src_input, src_target) in enumerate(src_dataloader):

            src_input = src_input.float().cuda()
            src_target = src_target.long().cuda()
            alpha = torch.tensor([0.0]).cuda()
            src_cls_p, _, _ = model(src_input, alpha=alpha, is_source=True)
            s_acc = s_cls_metric(F.softmax(src_cls_p, dim=-1).argmax(-1).cpu(), src_target.reshape(-1).cpu())

        out_dict["src_acc"] = s_cls_metric.compute()  

        for i, (idx, tar_input, tar_target) in enumerate(tar_dataloader):

            tar_input = tar_input.float().cuda()
            tar_target = tar_target.long().cuda()

            tar_cls_p, _, _ = model(tar_input, alpha=alpha, is_source=args.use_domain_bn is False)
            t_acc = t_cls_metric(F.softmax(tar_cls_p, dim=-1).argmax(-1).cpu(), tar_target.reshape(-1).cpu())

        out_dict["tar_acc"] = t_cls_metric.compute()  

    
    return out_dict
