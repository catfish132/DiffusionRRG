# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import tqdm
import os
import pickle

import time
import torch
from .defaults import DefaultTrainer
from xmodaler.config import kfg
from xmodaler.functional import bits_to_decimal
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY
from .rrg_d import RRG_D
from torch.autograd import Variable

__all__ = ['BitDiffusionGanTrainer']


@ENGINE_REGISTRY.register()
class BitDiffusionGanTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(BitDiffusionGanTrainer, self).__init__(cfg)
        self.debug = cfg.DEBUG
        # 设置判别器
        self.D = RRG_D(cfg)
        self.D = self.D.cuda()
        self.D_optimizer = self.build_optimizer(cfg, self.D)
        self.D_loss = torch.nn.BCELoss()
        self.D_bp_period = 2
        self.D_cur_period = 0

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter // self.iters_per_epoch)
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)

        data_time = time.perf_counter() - start
        data = comm.unwrap_model(self.model).preprocess_batch(data)
        if self.debug:
            origin_targets = data[kfg.U_TOKENS_IDS]
            origin_targets_str = self.model.beam_searcher.output_sents(origin_targets.view(-1, 20))

        batch_size = data['U_TOKENS_IDS'].size(0)
        real_label = Variable(torch.ones(batch_size)).cuda()
        fake_label = Variable(torch.zeros(batch_size)).cuda()
        outputs_dict = self.model(data)
        if self.D_cur_period == self.D_bp_period:
            ########################## 更新判别器
            for p in self.D.parameters():
                p.requires_grad = True
            # compute loss of fake_img
            fake_report = outputs_dict['U_LOGITS']  # 噪声输入G中获得假冒的报告
            fake_report = fake_report.clone().detach()  # 将fake report变成叶子节点，这样就避免了更新生成模型
            D_fake_logit = self.D(fake_report).squeeze()  # D　判断假冒报告的值 #
            D_loss_fake = self.D_loss(D_fake_logit, fake_label)  # 计算 假冒报告判别结果与0的损失，这样判别器就能使得真实图片的得分尽量高，假冒图片的得分尽量低

            # compute loss of real_img
            real_reports = data['U_TARGET_IDS']
            real_reports = real_reports.clone().detach()
            real_noise = torch.normal(0., 0.05, size=real_reports.size()).cuda()
            real_reports = torch.clamp(real_reports + real_noise, -1., 1.)
            D_real_logit = self.D(real_reports).squeeze()
            # closer to 1 means better 。 真值输入D的结果与真实标签1对比，如果D输出的真值结果与1越接近，说明D可以识别真实值
            D_loss_real = self.D_loss(D_real_logit, real_label)

            # bp and optimize
            D_loss = D_loss_real + D_loss_fake
            self.D_optimizer.zero_grad()  # 训练D网络，分别设置优化器
            D_loss.backward()  # 反向传递D 的梯度，再更新D
            self.D_optimizer.step()
            for p in self.D.parameters():  # 更新G时需要冻住D
                p.requires_grad = False
            self.D_cur_period = 0
        else:
            self.D_cur_period += 1
        ############################ 更新主模型
        fake_report = outputs_dict['U_LOGITS']  # fake_report 重新指向模型生成的报告，使得可以更新生成模型
        if self.debug:
            image_ids = outputs_dict[kfg.IDS]
            targets_str = self.decode_bit_str(outputs_dict[kfg.U_TARGET_IDS])
            predict_str = self.decode_bit_str(outputs_dict[kfg.U_LOGITS])

            for image_id, o_target, target, pred in zip(image_ids, origin_targets_str, targets_str, predict_str):
                print("{}: \nPred:\t{}\nTarget\t{}\nGroundT\t{}\n".format(image_id, pred, target, o_target))

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)

        D_fake_logit = self.D(fake_report).squeeze()
        g_loss = self.D_loss(D_fake_logit, real_label)  # G生成的假冒图片和1算loss 这样就能更新生成器了
        losses_dict.update({'G_loss': g_loss})

        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        if self.D_cur_period == 0:
            losses_dict.update({'D_loss': D_loss})
            losses_dict.update({'D_loss_real': D_loss_real})
            losses_dict.update({'D_loss_fake': D_loss_fake})
        self._write_metrics(losses_dict, data_time)

        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)

    def decode_bit_str(self, input_ids_bit):
        target_ids = bits_to_decimal(input_ids_bit, vocab_size=10200, bits=14)
        target_ids = target_ids.clamp(0., 10199.).long()
        outputs_str = self.model.beam_searcher.output_sents(target_ids)
        return outputs_str

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        if cfg.DATALOADER.INFERENCE_TRAIN == False:
            return super().test(cfg, model, test_data_loader, evaluator, epoch)

        else:
            model.eval()
            results = {}
            with torch.no_grad():
                for data in tqdm.tqdm(test_data_loader):
                    data = comm.unwrap_model(model).preprocess_batch(data)
                    ids = data[kfg.IDS]

                    res = model(data, use_beam_search=True, output_sents=False)

                    g_sents_ids = res[kfg.G_SENTS_IDS]
                    # mask-out all words after the first [EOS]
                    eos_id = comm.unwrap_model(model).beam_searcher.eos_token_id
                    mask = (torch.cumsum((g_sents_ids == eos_id), dim=-1) == 0).long()
                    g_sents_ids = g_sents_ids * mask
                    g_sents_ids = g_sents_ids.cpu().numpy()

                    for id, g_sent_ids in zip(ids, g_sents_ids):
                        results[id] = g_sent_ids.reshape(1, -1)

            # save results in the output_dir
            if evaluator.output_dir is not None:
                filename = 'ep_{}_ts_{}_td_{}.pkl'.format(epoch, int(cfg.DECODE_STRATEGY.DIFFUSION.TIMESTEPS),
                                                          int(cfg.DECODE_STRATEGY.DIFFUSION.TIME_DIFFERENCE))
                file_path = os.path.join(evaluator.output_dir, filename)
                with open(file_path, "wb") as f:
                    pickle.dump(results, f, protocol=4)

            model.train()
            return ''
