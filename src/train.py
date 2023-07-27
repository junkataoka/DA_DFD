import torch
from helper import adjust_learning_rate

def train_batch(model, src_train_batch, tar_train_batch, 
                        src_train_dataloader, tar_train_dataloader, 
                        optimizer, criterion, cur_epoch, logger, args):

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

    loss_dict = {}
    adjust_learning_rate(optimizer, args.lr, cur_epoch, args.epochs) # adjust learning rate
    model.train()
    optimizer.zero_grad()

    src_class = model(src_input)
    tar_class = model(tar_input)
    src_class_logprob = torch.log_softmax(src_class, dim=-1)
    tar_class_logprob = torch.log_softmax(tar_class, dim=-1)
    loss_dict["src_loss"] = criterion(src_class_logprob, src_target)
    loss_dict["tar_loss"] = criterion(tar_class_logprob, tar_target)
    loss = loss_dict["src_loss"]
    loss.backward()
    optimizer.step()
    logger.log(loss_dict)
