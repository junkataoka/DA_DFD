import torchmetrics
import torch
from avatar import WAVATAR
from dataloader import generate_dataset


def validate(model, src_dataloader, tar_dataloader, num_classes, logger):
    model.eval()
    s_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    t_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    s_domain_metric = torchmetrics.Accuracy(task="binary", num_classes=2) 
    t_domain_metric = torchmetrics.Accuracy(task="binary", num_classes=2)

    out_dict = {
        "src_label": [],
        "tar_label": [],
        "tar_label_ps": [],

        "src_index": [],
        "tar_index": [],

        "src_center": None,
        "tar_center": None,

        "src_acc": None,
        "tar_acc": None,

        "src_feature": [],
        "tar_feature": [],
        }

    with torch.no_grad():
        for i, (idx, src_input, src_target) in enumerate(src_dataloader):

            src_input = src_input.float().cuda()
            src_target = src_target.long().cuda()
            src_cls_p, src_dis_p, src_feature = model(src_input)

            if i == 0:
                c_src = torch.cuda.FloatTensor(num_classes, src_feature.shape[-1]).fill_(0)
                count_s = torch.cuda.FloatTensor(num_classes, 1).fill_(0)
            
            src_onehot = torch.cuda.FloatTensor(src_cls_p.size()).fill_(0)
            src_onehot.scatter_(1, src_target, torch.ones(src_cls_p.size(0), 1).cuda())

            c_src += (src_feature.unsqueeze(1) * src_onehot.unsqueeze(2)).sum(0)
            count_s += src_onehot.sum(0).unsqueeze(1)

            s_acc = s_cls_metric(src_cls_p.argmax(-1).cpu(), src_target.reshape(-1).cpu())
            out_dict["src_label"].append(src_target.reshape(-1))
            out_dict["src_feature"].append(src_feature)
            out_dict["src_index"].append(idx)

        c_src /= count_s
        s_acc_all = s_cls_metric.compute()  
        out_dict["src_center"] = c_src
        out_dict["src_acc"] = s_acc_all

        for i, (idx, tar_input, tar_target) in enumerate(tar_dataloader):

            tar_input = tar_input.float().cuda()
            tar_target = tar_target.long().cuda()
            tar_cls_p, tar_dis_p, tar_feature = model(tar_input)

            if i == 0:
                c_tar = torch.cuda.FloatTensor(num_classes, tar_feature.shape[-1]).fill_(0)
                count_t = torch.cuda.FloatTensor(num_classes, 1).fill_(0)

            tar_onehot = torch.cuda.FloatTensor(tar_cls_p.size()).fill_(0)
            tar_pred = tar_cls_p.argmax(-1)
            tar_onehot.scatter_(1, tar_pred.unsqueeze(1), torch.ones(tar_cls_p.size(0), 1).cuda())

            c_tar += (tar_feature.unsqueeze(1) * tar_onehot.unsqueeze(2)).sum(0)
            count_t += tar_onehot.sum(0).unsqueeze(1)

            t_acc = t_cls_metric(tar_cls_p.argmax(-1).cpu(), tar_target.reshape(-1).cpu())
            out_dict["tar_label"].append(tar_target.reshape(-1))
            out_dict["tar_label_ps"].append(tar_pred)
            out_dict["tar_feature"].append(tar_feature)
            out_dict["tar_index"].append(idx)

        c_tar /= count_t
        out_dict["tar_center"] = c_tar
        out_dict["tar_acc"] = t_cls_metric.compute()  

    return {k: torch.concat(v, dim=0) if type(v) is list else v for k, v in out_dict.items()}

# @pytest.mark.skip(reason="no way of currently testing this")
def test_validate():
    # Test validate function

    num_classes = 4
    model = WAVATAR(1, num_classes).cuda()
    src_dataset, tar_dataset = generate_dataset("/data/home/jkataok1/DA_DFD/data/processed", 
                     "CWRU", "CWRU", 0, 1)
    src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=24, shuffle=True, )
    tar_dataloader = torch.utils.data.DataLoader(tar_dataset, batch_size=24)
    logger=None
    out_dict = validate(model, src_dataloader, tar_dataloader, num_classes, logger)

