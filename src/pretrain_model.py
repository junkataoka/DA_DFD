# import libraries
import torch
from models.reflownet import EncoderDecoderConvLSTM
import torch.nn as nn
import click
from data.dataloader import generate_dataloader
import os
import math
import wandb


def adjust_learning_rate(optimizer, epoch, epoch_size, lr):
    """Adjust the learning rate according the epoch"""
    lr = lr / math.pow((1 + 10 * epoch / epoch_size), 0.75)
    for param_group in optimizer.param_groups:
       if "linear" in param_group['name']:
           param_group['lr'] = lr
       elif "decoder" in param_group['name'] or "encoder" in param_group["name"]:
           param_group['lr'] = lr
       else:
           raise ValueError('The required parameter group does not exist.')

def pretrain(src_x, src_y, model, optimizer, criterions, num_areas):

    optimizer.zero_grad()
    src_output, _, _ = model(src_x, domain="src", future_step=num_areas)

    src_loss = criterions["mse"](src_output, src_y)

    loss = src_loss 
    loss.backward()
    optimizer.step()
    print(f"Loss: {src_loss}")
    return {"loss":loss, "src_loss": src_loss}

def validate(tar_x_test, tar_y_test, criterions, model, num_areas):
    with torch.no_grad():
        tar_output, _, _ = model(tar_x_test, domain="tar", future_step=num_areas)

        tar_loss = criterions["mse"](tar_output, tar_y_test)
        print(f"Tar Loss: {tar_loss}")
        return {"test_tar_loss": tar_loss}

@click.command()
@click.option('--n_hidden_dim', nargs=1, type=int, default=3)
@click.option('--lr', nargs=1, type=float, default=0.0005)
@click.option('--batch_size', nargs=1, type=int, default=48)
@click.option('--epoch_size', nargs=1, type=int, default=100)
@click.option('--channel', nargs=1, type=int, default=2)
@click.option('--seq_len', nargs=1, type=int, default=15)
@click.option('--num_areas', nargs=1, type=int, default=7)
@click.argument('data_path', nargs=1, type=click.Path(exists=True))
@click.option('--log', nargs=1, type=str, default="test0")
def main(n_hidden_dim, lr, batch_size, epoch_size, data_path, channel, seq_len, num_areas, log):
    # load model
    model = EncoderDecoderConvLSTM(nf=n_hidden_dim, in_chan=channel, seq_len=seq_len).double().cuda()

    if not os.path.exists(f"./models/{log}"):
        os.mkdir(f"./models/{log}")

    # load dataloaders
    src_dataloader = generate_dataloader(
        geom_path = os.path.join(data_path, "train-src-GEOM"),
        heatmap_path = os.path.join(data_path, "train-src-HEATMAP"),
        recipe_path = os.path.join(data_path, "train-src-RECIPE"),
        batch_size=batch_size,
        train=True
    )
    test_tar_dataloader = generate_dataloader(
        geom_path = os.path.join(data_path, "test-tar-GEOM"),
        heatmap_path = os.path.join(data_path, "test-tar-HEATMAP"),
        recipe_path = os.path.join(data_path, "test-tar-RECIPE"),
        batch_size=1,
        train=False
    )

    # define optimizer and hyperparameters
    linear_param = [i[1] for i in model.named_parameters() if "src" in i[0]]
    feat_param = [i[1] for i in model.named_parameters() if not "linear" in i[0]]
    optimizer = torch.optim.SGD(
        [{"params": linear_param, "name": "linear"},
         {"params": feat_param, "name": "encoderdecoder"}
         ], 
        lr=lr, momentum=0.9, weight_decay=1e-2)

    # define loss functions
    criterions = {"mse": nn.MSELoss()}

    run = wandb.init(project="reflownet", 
                     config={
                        "log":log,
                        "pretrain":True
                     })

    for epoch in range(epoch_size):

        run.log({"epoch": epoch+1})
        for i in range(len(src_dataloader)):
            src_x, src_y = next(iter(src_dataloader))
            src_x = src_x.double().cuda()
            src_y = torch.log(src_y).double().cuda()
            loss_dict_train = pretrain(src_x, src_y, model, optimizer, criterions, num_areas=num_areas) 

            run.log(loss_dict_train)
            run.log({"batch_percent": (i+1) * len(src_dataloader)})
        
        for i in range(len(test_tar_dataloader)):

            tar_x, tar_y = next(iter(test_tar_dataloader))
            tar_x = tar_x.double().cuda()
            tar_y = torch.log(tar_y).double().cuda()
            loss_dict_val = validate(tar_x, tar_y, criterions=criterions, model=model, num_areas=num_areas)

            run.log(loss_dict_val)

        adjust_learning_rate(optimizer, epoch, epoch_size, lr)

        
    torch.save(model.state_dict(), f"./models/{log}/pretrained.ckpt")
    run.finish()

if __name__ == '__main__':
    main()
