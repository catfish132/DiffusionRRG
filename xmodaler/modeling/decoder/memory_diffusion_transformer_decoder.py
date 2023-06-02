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
from ..layers.bert import BertGenerationLayer

__all__ = ["MemoryDiffusionTransformerDecoder"]


@DECODER_REGISTRY.register()
class MemoryDiffusionTransformerDecoder(TransformerDecoder):
    @configurable
    def __init__(
            self,
            *,
            num_generation_layers: int,
            bert_generation_layers,
            d_model,
            max_seq_len
    ):
        super(MemoryDiffusionTransformerDecoder, self).__init__(
            num_generation_layers=num_generation_layers,
            bert_generation_layers=bert_generation_layers
        )
        self.memory = nn.Parameter(torch.FloatTensor(1, max_seq_len, d_model))
        # IO
        # 插槽嵌入层
        self.memory_embeddings = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )

    @classmethod
    def from_config(cls, cfg):
        bert_generation_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_GENERATION_LAYERS)]
        )
        return {
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "bert_generation_layers": bert_generation_layers,
            'd_model': cfg.MODEL.DECODER_DIM,
            'max_seq_len': cfg.MODEL.MAX_SEQ_LEN
        }

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

        u_tfeats_arr = []
        u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
        ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]

        u_tfeats = self.memory_embeddings(torch.cat([u_tfeats, self.memory.repeat(u_tfeats.shape[0],1,1)], -1))

        for i, layer_module in enumerate(self.g_layers):
            u_tfeats = layer_module(u_tfeats, vfeats, None, None,
                                    t_history_states=None)  # ext_u_tmasks=None, ext_vmasks=None
            u_tfeats_arr.append(u_tfeats)
        ret.update({kfg.U_HIDDEN_STATES: u_tfeats_arr})

        return ret
