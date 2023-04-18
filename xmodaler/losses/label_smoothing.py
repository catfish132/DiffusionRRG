# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class LabelSmoothing(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            label_smoothing,
            eos_id,
            loss_weight,
    ):
        super(LabelSmoothing, self).__init__()
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - self.label_smoothing
        self.criterion = nn.KLDivLoss(reduction='none')
        self.eos_id = eos_id
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            "label_smoothing": cfg.LOSSES.LABELSMOOTHING,
            'eos_id': cfg.SCORER.EOS_ID,
            'loss_weight': cfg.LOSSES.CLS_LOSS_WEIGHT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def Forward(self, logits, targets, mask):
        logP = F.log_softmax(logits.view(-1, logits.shape[-1]), dim=-1)
        targets = targets.view(-1)
        mask = mask.view(-1)

        assign_seq = targets
        assign_seq[assign_seq < 0] = 0

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        # 本质上logsoftmax后，概率为0的多分类应当=负无穷，这里gt为了避免负无穷，用了最小值-9..
        # 对于命中的分类，gt应当设定为1，这里使用0.9
        # 本质上还是用kl散度来拟合（因为gt用的是概率，而不是硬标签）
        # 句子长度短说明模型有很长一段是不拟合ｇｔ的
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss = self.criterion(logP, true_dist).sum(1)
        loss = torch.masked_select(loss, mask).mean()
        return loss

    def forward(self, outputs_dict):
        ret = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.U_TOKENS_IDS]

            seq = outputs_dict[kfg.U_TOKENS_IDS]
            mask = (torch.cumsum((seq == self.eos_id), dim=-1) == 0)
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
            # len(mask)=len(seq)+1,添加了<EOS>
            loss = self.Forward(logits, targets, mask)
            ret.update({'LabelSmoothing(G) loss': loss * self.loss_weight})
        return ret
