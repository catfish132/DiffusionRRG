# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .transformer_decoder import TransformerDecoder
from .build import DECODER_REGISTRY
from ..layers.bert import CircleBertGenerationLayer

__all__ = ["CircleDiffusionTransformerDecoder"]

@DECODER_REGISTRY.register()
class CircleDiffusionTransformerDecoder(TransformerDecoder):
    @configurable
    def __init__(
        self,
        *,
        num_generation_layers: int,
        bert_generation_layers
    ):
        super(CircleDiffusionTransformerDecoder, self).__init__(
            num_generation_layers=num_generation_layers,
            bert_generation_layers=bert_generation_layers
        )
    @classmethod
    def from_config(cls, cfg):
        bert_generation_layers = nn.ModuleList(
            [CircleBertGenerationLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_GENERATION_LAYERS)]
        )
        return {
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "bert_generation_layers": bert_generation_layers,
        }

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

        u_tfeats_arr = []
        u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
        ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]

        for i, layer_module in enumerate(self.g_layers):
            u_tfeats,vfeats = layer_module(u_tfeats, vfeats, ext_u_tmasks, ext_vmasks, t_history_states=None)
            u_tfeats_arr.append(u_tfeats)
        ret.update({ kfg.U_HIDDEN_STATES: u_tfeats_arr })

        return ret