from avatar import WAVATAR
from ast_models import ASTModel
from torch import nn
import torch
from collections import OrderedDict

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_model(model_name, C_in, class_num, checkpoint=None, input_time_dim=65, input_freq_dim=18):

    if model_name == 'avatar':
        return WAVATAR(C_in, class_num)
    elif model_name == 'ast':
        model = ASTModel(label_dim=class_num, input_tdim=input_time_dim, input_fdim=input_freq_dim,
                         imagenet_pretrain=False, audioset_pretrain=False)
        if checkpoint:
            # order of keys in state_dict is same between model and checkpoint
            state_dict_temp = torch.load(checkpoint)
            model_params = model.state_dict()
            for name, param in state_dict_temp.items():
                if name in model_params:
                    model_params[name].copy_(param)
                    print("copy {}".format(name), sep="\r")
                else:
                    print("skip {}".format(name), sep="\r")
            model.load_state_dict(state_dict_temp, strict=False)

            #state_dict = model.state_dict()

            #for key1, key2 in zip(state_dict.keys(), state_dict_temp.keys()):
            #    state_dict[key1] = state_dict_temp[key2]
            #model.mlp_head=Identity()
        return model
            
    else:
        raise ValueError("Model name not supported.")
