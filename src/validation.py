import torchmetrics
import torch

def validate(model, src_dataloader, tar_dataloader, num_classes, logger):
    model.eval()
    s_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    t_cls_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    s_domain_metric = torchmetrics.Accuracy(task="binary", num_classes=2) 
    t_domain_metric = torchmetrics.Accuracy(task="binary", num_classes=2)

    with torch.no_grad():
        for i, (src_input, src_target) in enumerate(src_dataloader):
            batch_size = src_input.shape[0]
            src_input = src_input.unsqueeze(1).float().cuda()
            src_target = src_target.long().cuda()
            s_domain_label = torch.zeros(batch_size).long().cuda()

            src_output, src_domain_output = model(src_input, 1.0)
            s_acc = s_cls_metric(src_output.cpu(), src_target.cpu())
            s_domain_acc = s_domain_metric(src_domain_output.argmax(-1).cpu(), s_domain_label.cpu())

        s_acc_all = s_cls_metric.compute()  
        s_domain_acc_all = s_domain_metric.compute()  
        logger.log({'src_acc': s_acc_all})
        logger.log({'src_domain_acc': s_domain_acc_all})


        for i, (tar_input, tar_target) in enumerate(tar_dataloader):
            batch_size = tar_input.shape[0]
            t_domain_label = torch.ones(batch_size).long().cuda()
            tar_input = tar_input.unsqueeze(1).float().cuda()
            tar_target = tar_target.long().cuda()

            tar_output, tar_domain_output = model(tar_input, 1.0)
            t_acc = t_cls_metric(tar_output.cpu(), tar_target.cpu())
            t_domain_acc = t_domain_metric(tar_domain_output.argmax(-1).cpu(), t_domain_label.cpu())

        t_acc_all = t_cls_metric.compute()  
        t_domain_acc_all = t_domain_metric.compute()  
        logger.log({'tar_acc': t_acc_all})
        logger.log({'tar_domain_acc': t_domain_acc_all})

        
    return s_acc, t_acc

