import torch
import torch.optim as optim
import torch.nn as nn

from ..modeling.blocks import MaskedConv1D, Scale, AffineDropPath, LayerNorm, LayerScale


def make_optimizer(config, model):
    """ create optimizer return a supported optimizer """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm, nn.LayerNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath, LayerScale)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif pn.endswith('A_log') or pn.endswith("D_b") or pn.endswith("D") or pn.endswith("A_b_log") or pn.endswith("forward_embed") or pn.endswith("backward_embed"):
                # corner case for mamba
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )
    
    # create the pytorch optimizer object
    optimizer_name = config.get('name', None)
    optimizer_args = config.get(optimizer_name, {})
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_args['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    if optimizer_name in optim.__dict__:
        optimizer_class = optim.__dict__[optimizer_name]
        optimizer = optimizer_class(optim_groups, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer