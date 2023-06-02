# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from .mscoco import MSCoCoDataset
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor
from ..build import DATASETS_REGISTRY

__all__ = ["RRGDiffusionDataset"]


@DATASETS_REGISTRY.register()
class RRGDiffusionDataset:
    @configurable
    def __init__(
            self,
            stage: str,
            anno_folder: str,
            similar_path: str,
            max_seq_len: int,
            image_path: str,
            cas_rand_ratio,
            dataset_name: str
    ):
        self.stage = stage
        self.anno_folder = anno_folder  # 我们直接使用anno_folder 指向标注文件
        self.similar_path = similar_path
        self.max_seq_len = max_seq_len
        self.image_path = image_path
        self.cas_rand_ratio = cas_rand_ratio
        if stage == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.similar_file = np.load(self.similar_path, allow_pickle=True).item()
        self.dataset_name = dataset_name

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = {"stage": stage,
               "anno_folder": cfg.DATALOADER.ANNO_FOLDER,
               "similar_path": cfg.DATALOADER.SIMILAR_PATH,
               "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
               "image_path": cfg.DATALOADER.IMAGE_PATH,
               "cas_rand_ratio": cfg.DATALOADER.CASCADED_SENT_RAND_RATIO,
               'dataset_name': cfg.DATASETS.NAME
               }
        return ret

    def load_data(self, cfg):

        if self.stage == 'test' and cfg.DATALOADER.INFERENCE_TRAIN == True:
            datalist = []
            for split in ['train', 'test']:
                anno_file = self.anno_folder.format(split)  # 需要修改
                tmp_datalist = pickle.load(open(anno_file, 'rb'), encoding='bytes')
                datalist.extend(tmp_datalist)
        else:
            datalist = pickle.load(open(self.anno_folder, 'rb'), encoding='bytes')
        datalist = datalist[self.stage]

        if len(cfg.DATALOADER.CASCADED_FILE) > 0:
            cascaded_pred = pickle.load(open(cfg.DATALOADER.CASCADED_FILE, 'rb'), encoding='bytes')
            for i in range(len(datalist)):
                image_id = str(datalist[i]['image_id'])
                datalist[i]['cascaded_tokens_ids'] = cascaded_pred[image_id]

        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        id = dataset_dict['id']
        image_list = dataset_dict['image_path']
        images = []
        for path in image_list:
            image = Image.open(os.path.join(self.image_path, path)).convert('RGB')
            image = self.transform(image)
            images.append(image)

        if self.stage != 'train':  # 如果是推理模式就不需要提供token了，计算指标可能在后续会进行
            u_tokens_ids = np.array(dataset_dict['report'], dtype=np.int64)
            u_tokens_type = np.zeros(self.max_seq_len, dtype=np.int64)
        else:
            u_tokens_ids = np.array(dataset_dict['report'], dtype=np.int64)[:self.max_seq_len]
            u_tokens_ids[-1] = 0
            u_tokens_type = np.zeros((len(u_tokens_ids)), dtype=np.int64)
        # 查找相似报告的特征
        if self.dataset_name == 'MIMIC_CXR':
            similar = [self.similar_file[similar_id] for similar_id in dataset_dict['similar']]
            ret = {
                kfg.IDS: id,
                kfg.U_TOKENS_IDS: u_tokens_ids,
                kfg.U_TOKENS_TYPE: u_tokens_type,
                kfg.IMAGES: images,
                kfg.SIMILAR: similar
            }
        else:
            ret = {
                kfg.IDS: id,
                kfg.U_TOKENS_IDS: u_tokens_ids,
                kfg.U_TOKENS_TYPE: u_tokens_type,
                kfg.IMAGES: images,
                # kfg.SIMILAR: similar
            }

        if 'cascaded_tokens_ids' in dataset_dict:
            cascaded_tokens_ids = dataset_dict['cascaded_tokens_ids']
            if cascaded_tokens_ids.shape[0] == 1:
                cascaded_tokens_ids = cascaded_tokens_ids.reshape(-1)

            if self.stage == 'train':
                ret[kfg.C_TOKENS_IDS] = [cascaded_tokens_ids for _ in range(self.seq_per_img)]
            else:
                ret[kfg.C_TOKENS_IDS] = [cascaded_tokens_ids]

        if self.stage != 'train':
            dict_as_tensor(ret)
            return ret

        if kfg.C_TOKENS_IDS in ret and self.cas_rand_ratio > 0.0:
            # rand replace augmentation for cascaded_tokens_ids
            new_cascaded_tokens_ids_list = []
            for cascaded_tokens_ids in ret[kfg.C_TOKENS_IDS]:
                for i in range(len(cascaded_tokens_ids)):
                    if cascaded_tokens_ids[i] == 0:
                        break

                    if random.random() < self.cas_rand_ratio:
                        # random replace the word in the i-th place
                        rand_word = random.randint(1, 10198)  # Return a random integer N such that a <= N <= b
                        cascaded_tokens_ids[i] = int(rand_word)
                new_cascaded_tokens_ids_list.append(cascaded_tokens_ids)
            ret[kfg.C_TOKENS_IDS] = new_cascaded_tokens_ids_list

        dict_as_tensor(ret)
        return ret
