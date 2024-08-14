import torch
from .blocks import *

class ResidualSharedBiMambaBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_module = EmbeddingModule(**kwargs.get("EmbeddingModule", {}))
        self.stem_module = StemModule(**kwargs.get("StemModule", {}))
        self.branch_module = BranchModule(**kwargs.get("BranchModule", {}))
        
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    torch.nn.init.constant_(module.bias, 0.)
    
    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        
        # embedding network
        x, mask = self.embedding_module(x, mask)
        x, mask = self.stem_module(x, mask)
        x, mask = self.branch_module(x, mask)
        
        return x, mask
