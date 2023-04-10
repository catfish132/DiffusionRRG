# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import pickle
import sys
import tempfile
import json
from json import encoder
from xmodaler.config import kfg
from xmodaler.config import configurable
from xmodaler.utils import comm
from .build import EVALUATION_REGISTRY

sys.path.append(kfg.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from .metrics import compute_scores


@EVALUATION_REGISTRY.register()
class RRGEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(RRGEvaler, self).__init__()
        self.cfg = cfg
        self.metrics = compute_scores
        self.id2gt = pickle.load(open(annfile, 'rb'))['test']
        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir) and comm.is_main_process():
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

    def eval(self, results, epoch):
        # results = {cfg.INFERENCE.ID_KEY:id,cfg.INFERENCE.VALUE:output}
        if self.output_dir is not None:
            json.dump(results, open(os.path.join(self.output_dir, str(epoch) + '.json'), "w"))
        reports, gts = {}, {}
        for sample in results:
            id = sample[self.cfg.INFERENCE.ID_KEY]
            report = [sample[self.cfg.INFERENCE.VALUE]]
            gt = [self.id2gt[id]]
            reports[id] = report
            gts[id] = gt

        # batch_gt = {id: self.id2gt[id] for id in results[self.cfg.INFERENCE.ID_KEY]}
        # output = {id: out for id, out in zip(results[self.cfg.INFERENCE.ID_KEY], results[self.cfg.INFERENCE.VALUE])}
        val_out = self.metrics(gts, reports)
        return val_out
