# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class WeightCrossEntropy(nn.Module):
    @configurable
    def __init__(self, vocab_size, eos_wight, eos_id, loss_weight):
        super(WeightCrossEntropy, self).__init__()
        weight = torch.ones((vocab_size,)).cuda()
        weight[0] = eos_wight
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
        self.eos_id = eos_id
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            'vocab_size': cfg.MODEL.VOCAB_SIZE,
            'eos_wight': cfg.LOSSES.EOS_WEIGHT,
            'eos_id': cfg.SCORER.EOS_ID,
            'loss_weight': cfg.LOSSES.CLS_LOSS_WEIGHT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.U_TOKENS_IDS]

            seq = outputs_dict[kfg.U_TOKENS_IDS]
            mask = (torch.cumsum((seq == self.eos_id), dim=-1) == 0)
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

            logits = logits[mask, :]
            targets = targets[mask].long()
            loss = self.criterion(logits, targets)
            ret.update({'CrossEntropy loss(G)': loss * self.loss_weight})

        return ret
