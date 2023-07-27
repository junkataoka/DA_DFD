# %%
import torch.utils.data as Data
from helper import (count_batch_on_large_dataset, weight_init, batch_norm_init, 
                    get_params, compute_threthold, compute_weights)
from avatar import WAVATAR
from validation import validate
from dataloader import generate_dataset
from train import train_avatar_batch
from kernel_kmeans import kernel_k_means_wrapper
import torch
import os
import wandb
import argparse
from construct_model import get_model

parser = argparse.ArgumentParser(description='DA_DFD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=48, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='epochs')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
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
    #model = WAVATAR(C_in=1, class_num=args.num_classes).to(args.device)
    #model.apply(weight_init)
    #model.apply(batch_norm_init)
    model = get_model(model_name=f"{args.model}", C_in=1, class_num=args.num_classes, checkpoint="/data/home/jkataok1/DA_DFD/src_models/audio_mdl.pth")

    if args.pretrained:
        print("load pretrained model from {}".format(args.source_model_path))
        src_model_name = f"src_{args.model}.pth"
        model_name = f"adapted_{args.model}.pth"
        model.load_state_dict(torch.load(os.path.join(args.source_model_path, "_".join([args.src_data, args.src_domain, args.tar_data, args.tar_domain, src_model_name]))))
    else:
        print("Pretraining model from scratch")
        model_name = f"src_{args.model}.pth"

    # define optimizer 
    params_enc = get_params(model, ["net", "fc"])
    params_cls = get_params(model, ["classifier"])
    optimizer_dict = {
        "encoder": torch.optim.SGD(params_enc,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay),
        "classifier": torch.optim.SGD(params_cls,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)}

    # Count batch size
    batch_count = count_batch_on_large_dataset(src_train_dataloader, tar_train_dataloader)
    # Count total iteration
    num_itern_total = args.epochs * batch_count
    # Initialize epoch
    epoch = 0
    count_itern_each_epoch = 0
    best_acc = 0.0
    criterion = torch.nn.NLLLoss()

    for itern in range(num_itern_total):
        src_train_batch = enumerate(src_train_dataloader)
        tar_trai_batch = enumerate(tar_train_dataloader)

        if (itern==0 or count_itern_each_epoch==batch_count):

            # Validate and compute source and target domain accuracy, and cluster center
            val_dict = validate(model, src_val_dataloader, tar_val_dataloader, args.num_classes, wandb)

            # Comptue target cluster center
            
            val_dict["tar_center"], val_dict["tar_label_kmeans"], val_dict["tar_acc_cluster"] = kernel_k_means_wrapper(val_dict["tar_feature"], 
                                                                   val_dict["tar_label_ps"], 
                                                                   val_dict["tar_label"], 
                                                                   epoch, args, best_prec=1e-4)

            val_dict["tar_weights"] = compute_weights(val_dict["tar_feature"], 
                                          val_dict["tar_label_ps"], 
                                          val_dict["tar_center"])

            val_dict["src_weights"]= compute_weights(val_dict["src_feature"], 
                                          val_dict["src_label"], 
                                          val_dict["src_center"])

            #val_dict["tar_label_ps_ord"] = val_dict["tar_label_kmeans"][val_dict["tar_index"]]

            val_dict["th"] = compute_threthold(val_dict["tar_weights"], 
                                               val_dict["tar_label_ps"], 
                                               args.num_classes)
            
            wandb.log({"src_acc": val_dict["src_acc"], 
                       "tar_acc": val_dict["tar_acc"], 
                       "cluster_acc": val_dict["tar_acc_cluster"],
                       "epoch": epoch})

            if val_dict["tar_acc"] > best_acc:
                best_acc = val_dict["tar_acc"]
                wandb.log({"best_acc": best_acc, "epoch": epoch})

                if args.pretrained:
                    torch.save(model.state_dict(), os.path.join(log, model_name))
                else:
                    torch.save(model.state_dict(), os.path.join(args.source_model_path, "_".join([args.src_data,args.src_domain, args.tar_data, args.tar_domain, model_name])))

            if itern != 0:
                count_itern_each_epoch = 0
                epoch += 1

        train_avatar_batch(model=model, src_train_batch=src_train_batch, tar_train_batch=tar_trai_batch, 
                            src_train_dataloader=src_train_dataloader, tar_train_dataloader=tar_train_dataloader, 
                            optimizer_dict=optimizer_dict, cur_epoch=epoch, logger=wandb, val_dict=val_dict, args=args)

        count_itern_each_epoch += 1

if __name__ == "__main__":
    main(args)


# %%
