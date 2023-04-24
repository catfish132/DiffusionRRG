# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import numpy as np
import pickle

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import SCORER_REGISTRY
from pycocoevalcap.bleu.bleu import Bleu
from collections import OrderedDict

__all__ = ['Bleu4Scorer']


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


@SCORER_REGISTRY.register()
class Bleu4Scorer(object):
    @configurable
    def __init__(
            self,
            *,
            weights,
            gt_path,
            eos_id
    ):
        self.scorers = [Bleu(4)]
        self.eos_id = eos_id
        self.weights = weights
        self.id2gt = pickle.load(open(gt_path, 'rb'))['train']
        self.types = ['Bleu-4']

    @classmethod
    def from_config(cls, cfg):
        return {
            'weights': cfg.SCORER.WEIGHTS,
            'gt_path': cfg.SCORER.GT_PATH,
            'eos_id': cfg.SCORER.EOS_ID
        }

    def get_sents(self, sent):
        words = []
        for word in sent:
            if word == self.eos_id:
                words.append(self.eos_id)
                break
            words.append(word)
        return words

    def __call__(self, batched_inputs):
        ids = batched_inputs[kfg.IDS]
        res = batched_inputs[kfg.G_SENTS_IDS]
        res = res.cpu().tolist()

        hypo = [self.get_sents(r) for r in res]
        reports, gts = {}, {}
        for id, report in zip(ids, hypo):
            reports[id] = [array_to_str(report)]
            gts[id] = [array_to_str(self.id2gt[id])]

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, reports, verbose=0)
            score = score[-1]
            scores = np.array(scores[-1])
            rewards += self.weights[i] * scores
            rewards_info[self.types[i]] = score
        rewards_info.update({kfg.REWARDS: rewards})
        return rewards_info
