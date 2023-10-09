# %%
import torch.utils.data as Data
from helper import (count_batch_on_large_dataset, weight_init, batch_norm_init, 
                    get_params, compute_threthold, compute_weights, define_param_groups,
                    copy_domain_batch_norm)
from validation import validate
from dataloader import generate_dataset
from train import train_batch
from kernel_kmeans import kernel_k_means_wrapper
import torch
import os
import wandb
import argparse
from construct_model import get_model
from losses import ContrastiveLoss

parser = argparse.ArgumentParser(description='DA_DFD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=48, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='epochs')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=1e-2, help='weight decay')
parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
parser.add_argument('--device', type=str, default="cuda:0", help='cuda device')
parser.add_argument('--data_path', type=str, default="/data/home/jkataok1/DA_DFD/data/processed", help='data path')
parser.add_argument('--src_data', type=str, default="CWRU", help='source data')
parser.add_argument('--tar_data', type=str, default="CWRU", help='target data')
parser.add_argument('--src_domain', type=str, default="0", help='source domain')
parser.add_argument('--tar_domain', type=str, default="1", help='target domain')
parser.add_argument('--log', type=str, default="log", help='log')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--source_model_path', type=str, default="src_models", help='source model path')
parser.add_argument('--warmup_epoch', type=int, default=1, help='warm up epoch size')
parser.add_argument('--model', type=str, default="ast", help='model name')
parser.add_argument('--accum_iter', type=int, default=1, help='accumulation of iteration to comptue gradient')
parser.add_argument('--input_channel', type=int, default=1, help='number of input channels')
parser.add_argument('--input_time_dim', type=int, default=65, help='number of time dimension')
parser.add_argument('--input_freq_dim', type=int, default=18, help='number of frequency dimension')
parser.add_argument('--use_domain_bn', action='store_true')
parser.add_argument('--use_domain_adv', action='store_true')
parser.add_argument('--use_tar_entropy', action='store_true')
parser.add_argument('--use_contra_learn', action='store_true')

args = parser.parse_args()


def main(args):

    # create log directory
    log = args.log + "/" + f"{args.src_data}" + f"{args.src_domain}" + "to" + f"{args.tar_data}" + f"{args.tar_domain}" \
                + "_lr" + f"{str(args.lr)}" + "_e" + f"{str(args.epochs)}" + "_b" + f"{str(args.batch_size)}"

    # create log directory
    if not os.path.isdir(log):
        os.makedirs(log)
        print("create new log directory")

    # initialize wandb
    hyperparameter_defaults = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        src_domain=args.src_domain,
        tar_domain=args.tar_domain,
        is_pretriained= True if args.pretrained else False,
        use_domain_bn=True if args.use_domain_bn else False,
        use_domain_adv=True if args.use_domain_adv else False,
        use_contra_learn=True if args.use_contra_learn else False
    )
    wandb.init(config=hyperparameter_defaults, name=log, project="DA_DFD")
    wandb.define_metric("src_acc", summary="max")
    wandb.define_metric("tar_acc", summary="max")

    # generate dataset for training
    src_dataset, tar_dataset = generate_dataset(args.data_path, args.src_data, args.tar_data, args.src_domain, args.tar_domain)
    # convert dataset to dataloader
    src_train_dataloader = Data.DataLoader(src_dataset, 
                                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    tar_train_dataloader = Data.DataLoader(tar_dataset,
                                        batch_size=args.batch_size, shuffle=True, drop_last=True) 

    src_val_dataloader = Data.DataLoader(src_dataset, 
                                        batch_size=args.batch_size, shuffle=True, drop_last=False)
    tar_val_dataloader = Data.DataLoader(tar_dataset,
                                        batch_size=args.batch_size, shuffle=True, drop_last=False) 
    
    # define model 
    model = get_model(model_name=f"{args.model}", C_in=args.input_channel, 
                      class_num=args.num_classes,
                      input_time_dim=args.input_time_dim,
                      input_freq_dim=args.input_freq_dim)

    model_name = ""

    if args.pretrained:
        print("load pretrained model from {}".format(args.source_model_path))
        model.load_state_dict(torch.load(os.path.join(args.source_model_path, "_".join([args.src_data, args.src_domain, args.tar_data, args.tar_domain, args.model + ".pth"]))))
        model_name += "da_"

    
    if args.use_domain_bn:
        print("Using domain batch normalization")
        model_name += "domain_bn_"
    
    if args.use_domain_adv:
        print("Using domain adversarial")
        model_name += "adv_"

    if args.use_contra_learn:
        print("Using contrastive learning")
        model_name += "contra_"

    model_name += f"{args.model}.pth"

    # Define model name


    model = model.to(args.device)
    # define optimizer 
    params = model.parameters()
    #params_enc = get_params(model, ["net", "fc"])
    #params_cls = get_params(model, ["classifier"])
    #optimizer_dict = {
    #    "encoder": torch.optim.SGD(params_enc,
    #                            lr=args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay),
    #    "classifier": torch.optim.SGD(params_cls,
    #                            lr=args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)}
    #optimizer = torch.optim.SGD(params,
    #                            lr=args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Count batch size
    batch_count = count_batch_on_large_dataset(src_train_dataloader, tar_train_dataloader)
    # count total iteration
    num_itern_total = args.epochs * batch_count
    # initialize epoch
    epoch = 0
    count_itern_each_epoch = 0
    best_acc = 0.0
    criterion = torch.nn.NLLLoss()
    criterion_contra = ContrastiveLoss(args.batch_size, temperature=0.5)

    for itern in range(num_itern_total):
        src_train_batch = enumerate(src_train_dataloader)
        tar_train_batch = enumerate(tar_train_dataloader)

        if (itern==0 or count_itern_each_epoch==batch_count):

            # Validate and compute source and target domain accuracy, and cluster center
            val_dict = validate(model, src_val_dataloader, tar_val_dataloader, args)

            wandb.log({"src_acc": val_dict["src_acc"], 
                       "tar_acc": val_dict["tar_acc"], 
                       "epoch": epoch})

            if not args.pretrained:
                acc_name = "tar_acc"
            else:
                acc_name = "tar_acc"

            if val_dict[acc_name] > best_acc:
                best_acc = val_dict[acc_name]
                wandb.log({"best_acc": best_acc, "epoch": epoch})

                if args.pretrained:
                    torch.save(model.state_dict(), os.path.join(log, model_name))
                else:
                    torch.save(model.state_dict(), os.path.join(args.source_model_path, "_".join([args.src_data,args.src_domain, args.tar_data, args.tar_domain, args.model + ".pth"])))
                    
            if itern != 0:
                count_itern_each_epoch = 0
                epoch += 1
        
        is_backprop = count_itern_each_epoch % args.accum_iter == 0

        train_batch(model=model, src_train_batch=src_train_batch, tar_train_batch=tar_train_batch, 
                            src_train_dataloader=src_train_dataloader, tar_train_dataloader=tar_train_dataloader, 
                            optimizer=optimizer, criterion=criterion, criterion_contra=criterion_contra, cur_epoch=epoch, 
                            logger=wandb, args=args, backprop=is_backprop)

        count_itern_each_epoch += 1

if __name__ == "__main__":
    main(args)
# %%
