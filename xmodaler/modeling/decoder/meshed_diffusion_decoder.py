# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.multihead_attention import MultiHeadAttention
from ..layers.positionwise_feedforward import PositionWiseFeedForward

from .decoder import Decoder
from .build import DECODER_REGISTRY

import numpy as np

__all__ = ["MeshedDiffusionDecoder"]

class MeshedDecoderLayer(nn.Module):
    def __init__(
        self, 
        *,
        d_model=512, 
        num_head=8, 
        d_ff=2048, 
        dropout=.1,
        enc_layer_num=3,
    ):
        super(MeshedDecoderLayer, self).__init__()

        d_k = d_v = d_model // num_head

        self.self_att = MultiHeadAttention( d_model=d_model, 
                                            d_k=d_k, 
                                            d_v=d_v, 
                                            num_head=num_head, 
                                            dropout=dropout
                                        )
        self.enc_att = MultiHeadAttention(  d_model=d_model, 
                                            d_k=d_k, 
                                            d_v=d_v, 
                                            num_head=num_head, 
                                            dropout=dropout
                                        )
                                        
        self.pwff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.fc_alpha = nn.ModuleList()
        for _ in range(enc_layer_num):
            self.fc_alpha.append(nn.Linear(2 * d_model, d_model))

        # init fc_alpha weights
        for i in range(enc_layer_num):
            nn.init.xavier_uniform_(self.fc_alpha[i].weight)
            nn.init.constant_(self.fc_alpha[i].bias, 0)

    def forward(self, input, enc_output, mask_self_att, mask_enc_att, history_states=None):
        self_att = self.self_att(input, input, input, mask_self_att, history_states=history_states)

        # cal attention on each encoder layer then weighted sum
        enc_att = 0
        for i in range(len(self.fc_alpha)):
            enc_att_k = self.enc_att(self_att, keys=enc_output[:, i], values=enc_output[:, i], attention_mask=mask_enc_att)
            alpha_k = torch.sigmoid(self.fc_alpha[i](torch.cat([self_att, enc_att_k], -1)))
            enc_att += enc_att_k * alpha_k
        enc_att = enc_att / np.sqrt(len(self.fc_alpha))

        ff = self.pwff(enc_att)
        return ff

@DECODER_REGISTRY.register()
class MeshedDiffusionDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        d_model: int , 
        num_layer: int,  
        num_att_head: int, 
        d_ff: int, 
        dropout: float,
        padding_idx: int, # -1
        enc_layer_num: int
    ):
        super(MeshedDiffusionDecoder, self).__init__()

        self.num_layers = num_layer
        self.d_model = d_model
        self.num_att_head = num_att_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.layers = nn.ModuleList([
            MeshedDecoderLayer(
                d_model=self.d_model, 
                num_head=self.num_att_head, 
                d_ff=self.d_ff, 
                dropout=self.dropout,
                enc_layer_num=enc_layer_num
            ) for _ in range(self.num_layers)
        ])

    @classmethod
    def from_config(cls, cfg):
        return {
            "d_model": cfg.MODEL.MESHEDMEORY.DECODER.DIM_MODEL,
            "num_layer": cfg.MODEL.MESHEDMEORY.DECODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.MESHEDMEORY.DECODER.NUM_ATT_HEAD,
            "d_ff": cfg.MODEL.MESHEDMEORY.DECODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.MESHEDMEORY.DECODER.DROPOUT,
            "padding_idx": -1, # default
            "enc_layer_num": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_LAYER
        }

    @classmethod
    def add_config(cls, cfg):
        if not hasattr(cfg.MODEL, "MESHEDMEORY"):
            cfg.MODEL.MESHEDMEORY = CN()

        cfg.MODEL.MESHEDMEORY.DECODER = CN()
        cfg.MODEL.MESHEDMEORY.DECODER.DIM_MODEL = 512
        cfg.MODEL.MESHEDMEORY.DECODER.NUM_LAYER = 3
        cfg.MODEL.MESHEDMEORY.DECODER.DROPOUT = 0.1
        cfg.MODEL.MESHEDMEORY.DECODER.NUM_ATT_HEAD = 8
        cfg.MODEL.MESHEDMEORY.DECODER.DIM_FEEDFORWARD = 2048

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.ATT_FEATS]

        u_tfeats_arr = []
        u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
        for i, layer_module in enumerate(self.layers):
            u_tfeats = layer_module(u_tfeats, vfeats, None, None,None)
            u_tfeats_arr.append(u_tfeats)
        ret.update({kfg.U_HIDDEN_STATES: u_tfeats_arr})

        return ret