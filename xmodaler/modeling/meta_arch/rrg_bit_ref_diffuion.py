# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torchvision.models as models
from random import random

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import beta_linear_log_snr, alpha_cosine_log_snr, log_snr_to_alpha_sigma, right_pad_dims_to
from xmodaler.functional import pad_tensor, dict_to_cuda, flat_list_of_lists
from . import RrgBitDiffusion
from .transformer_enc_dec import TransformerEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["RrgBitRefDiffusion"]

from ..embedding import build_embeddings
from ..encoder import build_encoder


@META_ARCH_REGISTRY.register()
class RrgBitRefDiffusion(RrgBitDiffusion):
    """
        这是使用相似报告的bit Diffusion
    """

    @configurable
    def __init__(
            self,
            *,
            vocab_size,
            max_seq_len,
            token_embed,
            visual_embed,
            encoder,
            decoder,
            predictor,
            greedy_decoder,
            beam_searcher,
            v_predictor,

            log_snr,
            similar_encoder
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher,
            v_predictor=v_predictor,
            log_snr=log_snr
        )
        self.similar_embed = nn.Linear(1024, 512)
        self.similar_encoder = similar_encoder

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        tmp_cfg = cfg.clone()
        tmp_cfg.defrost()
        tmp_cfg.MODEL.BERT.NUM_HIDDEN_LAYERS = 3
        tmp_cfg.freeze()
        r_encoder = build_encoder(tmp_cfg)

        ret.update({
            'similar_encoder': r_encoder
        })

        return ret

    def reshape_feat(self, x):
        batch_size, feat_size, _, _ = x.shape
        return x.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

    def _forward(self, batched_inputs):
        inputs = batched_inputs
        att_feat = self.reshape_feat(self.visual_backbone(batched_inputs[kfg.IMAGES]))
        if kfg.IMAGES2 in batched_inputs:
            att_feat2 = self.reshape_feat(self.visual_backbone(batched_inputs[kfg.IMAGES2]))
            att_feat = torch.cat([att_feat, att_feat2], dim=1)
        att_feat, att_mask = pad_tensor(att_feat, padding_value=0, use_mask=True)
        batched_inputs[kfg.ATT_FEATS] = att_feat
        batched_inputs[kfg.ATT_MASKS] = att_mask.to(att_feat.device)

        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = self.visual_embed(batched_inputs)
        inputs.update(ve_out)

        if self.encoder is not None:
            encoder_out_v = self.encoder(inputs, mode='v')
            inputs.update(encoder_out_v)

        # similar embed
        similar_out = self.similar_embed(inputs['similar'])
        inputs['similar_embed'] = similar_out

        # convert txt to bit representation
        input_ids_bit = self.token_embed.get_bit_repr(inputs[kfg.U_TOKENS_IDS])
        inputs.update({
            kfg.U_TOKENS_IDS_BIT: input_ids_bit,
            kfg.U_TARGET_IDS: input_ids_bit
        })

        # noise sample
        corrupt_out = self.noise_sample(inputs)
        inputs.update(corrupt_out)
        # noised_input_ids_bit = corrupt_out[kfg.U_TOKENS_IDS]

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                inputs = self._diff_decoder_forward(inputs)
                self_cond = inputs[kfg.U_LOGITS].detach_()
                del inputs[kfg.U_LOGITS]
        inputs.update({kfg.SELF_COND: self_cond})

        inputs = self._diff_decoder_forward(inputs)
        return inputs

    def _diff_decoder_forward(self, inputs):
        te_out = self.token_embed(inputs)  # inputs may have self_cond
        inputs.update(te_out)
        '''
            1.拼接原bit token和similar
            2.encode
            3.截取原token对应的部分，更新原token
        '''
        seq_length = inputs[kfg.U_TOKENS_IDS].shape[-1]
        r_enc_inputs = {
            kfg.ATT_FEATS: torch.cat([inputs[kfg.U_TOKEN_EMBED], inputs['similar_embed']], dim=1),
            kfg.EXT_ATT_MASKS: None
        }
        # 这里ref与similar同义
        r_encoder_out = self.similar_encoder(r_enc_inputs, mode='v')
        ref_awared_u_token_embed = r_encoder_out[kfg.ATT_FEATS][:, :seq_length, :].contiguous()
        inputs.update({
            kfg.U_TOKEN_EMBED: ref_awared_u_token_embed
        })

        # predict and take gradient step
        decoder_out = self.decoder(inputs)
        inputs.update(decoder_out)

        # bert hidden_size -> bit dim
        tlogits = self.predictor(inputs)
        inputs.update(tlogits)

        return inputs

    def noise_sample(self, inputs):
        # corrupt bit repr, i.e. 14-dim vector
        batch_size = inputs[kfg.ATT_FEATS].shape[0]
        device = inputs[kfg.ATT_FEATS].device

        # sample random times
        times = torch.zeros((batch_size,), device=device).float().uniform_(0, 0.999)

        bit_token_embed = inputs[kfg.U_TOKENS_IDS_BIT]
        noise = torch.randn_like(bit_token_embed)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(bit_token_embed, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)  # 从 noise 那里去生成 alpha 和均值量

        noised_bit_token_embed = alpha * bit_token_embed + sigma * noise
        return {
            kfg.U_TOKENS_IDS_BIT: noised_bit_token_embed,
            kfg.TIME_STEP: noise_level
        }

    def preprocess_batch(self, batched_inputs):
        super_ret = super().preprocess_batch(batched_inputs)

        sample_per_sample = batched_inputs[0].get(kfg.SAMPLE_PER_SAMPLE, 1)
        ret = {}

        # 处理similar相关的数据预处理
        similar = [torch.stack(i['SIMILAR']) for i in batched_inputs]
        similar = torch.stack(similar)
        ret['similar'] = similar
        dict_to_cuda(ret)
        super_ret.update(ret)
        return super_ret
