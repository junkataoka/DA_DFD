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

    if model_name == 'wavatar':
        return WAVATAR(C_in, class_num)
    elif model_name == 'ast':
        model = ASTModel(label_dim=4, input_tdim=class_num, imagenet_pretrain=False, audioset_pretrain=False)
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
            model.module.mlp_head=Identity()
        return model
            
    else:
        raise ValueError("Model name not supported.")
