import wandb
import math


# def log_metric(loss_dict, run, tag="train"):
#     for key, val in loss_dict.items():
#         run[f"{tag}_{key}"].append(val)

def test_neptune():
    run = wandb.init(project="reflownet")
    loss_dict = {"a": 0.5, "b": 0.3}
    run.log(loss_dict)
    loss_dict = {"a": 0.9, "b": 0.23143}
    run.log(loss_dict)
    run.finish()

def adjust_learning_rate(epoch, epoch_size, lr):
    lr = lr / math.pow(1+10 * epoch / epoch_size, 0.75)
    return lr
