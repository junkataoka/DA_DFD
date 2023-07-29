from avatar import WAVATAR
from ast_models import ASTModel
from torch import nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_model(model_name, C_in, class_num, checkpoint=None):

    if model_name == 'avatar':
        return WAVATAR(C_in, class_num)
    elif model_name == 'ast':
        model = ASTModel(label_dim=class_num, input_tdim=2048, imagenet_pretrain=False, audioset_pretrain=False)
        if checkpoint:
            # order of keys in state_dict is same between model and checkpoint
            state_dict_temp = torch.load(checkpoint)
            state_dict = model.state_dict()
            for key1, key2 in zip(state_dict.keys(), state_dict_temp.keys()):
                state_dict[key1] = state_dict_temp[key2]
            model.mlp_head=Identity()
        return model
            
    else:
        raise ValueError("Model name not supported.")
