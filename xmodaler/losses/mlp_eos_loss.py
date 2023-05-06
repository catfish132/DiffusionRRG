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
class MlpEosLoss(nn.Module):
    @configurable
    def __init__(self):
        super(MlpEosLoss, self).__init__()
        self.eos_criterion = nn.BCEWithLogitsLoss()

    @classmethod
    def from_config(cls, cfg):
        return {

        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret = {}
        if kfg.G_LOGITS in outputs_dict:
            eos_logit = outputs_dict['eos_logit']
            targets = outputs_dict[kfg.U_TOKENS_IDS]
            eos_targets = torch.zeros_like(targets).float()
            eos_targets[targets == 0] = 1.  # 即EOS为1其他都为0

            eos_loss = self.eos_criterion(eos_logit, eos_targets)

            ret.update({'Eos Loss:eos_loss': eos_loss})

        return ret
