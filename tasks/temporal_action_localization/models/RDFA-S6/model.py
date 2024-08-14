from omegaconf import DictConfig
from torch import nn
from libs.utils import util_system
from libs.modeling.backbones import *
from libs.modeling.necks import *
from libs.modeling.generators import *
from libs.modeling.heads import *


@util_system.logger_wraps()
class Model(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        """ Build backbone """
        backbone_info = cfg.model.backbone_info
        BackboneClass = globals().get(backbone_info.name, None)
        backbone_args = backbone_info.get(backbone_info.name, {})
        fpn_strides = [2**i for i in range(backbone_args.BranchModule.block_n + 1)]
        if backbone_args.get('mha_win_size', None):
            if isinstance(backbone_args.mha_win_size, int):
                backbone_args.mha_win_size = [backbone_args.mha_win_size]*len(fpn_strides)
            else:
                assert len(backbone_args.mha_win_size) == len(fpn_strides)
                backbone_args.mha_win_size = backbone_args.mha_win_size
            for stride, window_size in zip(fpn_strides, backbone_args.mha_win_size):
                effective_stride = stride * (window_size // 2) * 2 if window_size > 1 else stride
                assert backbone_args.max_seq_len % effective_stride == 0, "max_seq_len must be divisible by effective stride"
                # Update max_div_factor in loader settings if current stride is larger
                if cfg.dataset.loader.max_div_factor < effective_stride:
                    cfg.dataset.loader.max_div_factor = effective_stride
        else:
            for stride in fpn_strides:
                if cfg.dataset.loader.max_div_factor < stride:
                    cfg.dataset.loader.max_div_factor = stride
        self.backbone = BackboneClass(**backbone_args)
        
        """ Build neck """
        neck_info = cfg.model.neck_info
        NeckClass = globals().get(neck_info.name, None)
        neck_args = neck_info.get(neck_info.name, {})
        neck_args.in_channels = [neck_args.in_channels] * (backbone_args.BranchModule.block_n + 1)
        self.neck = NeckClass(**neck_args)
        
        """ Build generator """
        generator_info = cfg.model.generator_info
        GeneratorClass = globals().get(generator_info.name, None)
        generator_args = generator_info.get(generator_info.name, {})
        assert len(fpn_strides) == len(generator_args.regression_range)
        generator_args.fpn_levels = len(fpn_strides)
        self.generator = GeneratorClass(**generator_args)
        
        """ Build heads """
        head_info = cfg.model.head_info
        ClsHeadClass, RegHeadClass = [globals().get(head_name, None) for head_name in head_info.name]
        cls_head_args, reg_head_args = [head_info.get(head_name, {}) for head_name in head_info.name]
        reg_head_args.fpn_levels = len(fpn_strides)
        self.cls_head, self.reg_head = ClsHeadClass(**cls_head_args), RegHeadClass(**reg_head_args)
    
    def forward(self, batched_inputs, batched_masks):
        
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        # B=2, C=3200, T=2304
        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        
        # fpn_feats [16, 256, 768] ..[16, 256, 384]..[16, 256, 24]
        fpn_feats, fpn_masks = self.neck(feats, masks)
        
        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.generator(fpn_feats)
        
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        
        # permute the outputs
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        
        
        return points, fpn_masks, out_cls_logits, out_offsets