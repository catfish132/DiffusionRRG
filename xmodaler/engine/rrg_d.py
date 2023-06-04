import torch
import torch.nn as nn
from ..modeling.layers.bert import BertLayer

class RRG_D(nn.Module):
    def __init__(self, cfg):
        super(RRG_D, self).__init__()
        if cfg.MODEL.VOCAB_SIZE == 761:
            voc_size = 10
        else:
            voc_size = 13
        self.linear = nn.Linear(voc_size, 512)
        self.bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(3)]
        )
        self.predictor = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, input):
        h = self.linear(input)
        for layer_module in self.bert_layers:
            h, _ = layer_module(h,None)
        h = torch.mean(h, dim=-2)
        h = self.predictor(h)
        return h
