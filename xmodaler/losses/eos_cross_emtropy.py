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
class EosCrossEntropy(nn.Module):
    @configurable
    def __init__(self, vocab_size, eos_wight, eos_id, loss_weight):
        super(EosCrossEntropy, self).__init__()
        weight = torch.ones((vocab_size,)).cuda()
        weight[0] = eos_wight
        self.seq_criterion = nn.CrossEntropyLoss(weight=weight)
        self.eos_criterion = nn.BCEWithLogitsLoss()
        self.eos_id = eos_id
        self.seq_weight = loss_weight

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
            eos_targets = torch.zeros_like(seq).float()
            eos_targets[mask == False] = 1.  # 即EOS为1其他都为0
            eos_logits = logits[:, :, 0]  # 为EOS 维度单独做一个二分类，并且在gt=EOS之后的所有token都设置为EOS
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

            logits = logits[mask, :]
            targets = targets[mask].long()
            seq_loss = self.seq_criterion(logits, targets)
            # eos_logits = logits[:, 0]  # 如果仅对最后一个token计算loss的话
            # eos_targets = torch.zeros_like(targets).float()
            # eos_targets[targets == 0] = 1.
            eos_loss = self.eos_criterion(eos_logits, eos_targets)

            ret.update({'CrossEntropy loss(G)': seq_loss * self.seq_weight,
                        'Eos Loss:eos_loss': eos_loss})

        return ret
