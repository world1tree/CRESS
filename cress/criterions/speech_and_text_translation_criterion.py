# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig, 
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechAndTextTranslationCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mt_finetune: bool = field(
        default=False,
        metadata={"help": "st + mt multi-task finetune"},
    )
    cmlm_mask_ratio: float = field(
        default=0.15,
        metadata={"help": "mask ratio for cmlm, default=0.15"},
    )
    cmlm_weight: float = field(
        default=1.0,
        metadata={"help": "cmlm weight, default=1.0"},
    )

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion(
    "speech_and_text_translation", dataclass=SpeechAndTextTranslationCriterionConfig
)
class SpeechAndTextTranslationCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        mt_finetune=False,
        cmlm_mask_ratio=0.15,
        cmlm_weight=1.0,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.mt_finetune = mt_finetune
        self.cmlm_mask_ratio = cmlm_mask_ratio
        self.cmlm_weight = cmlm_weight

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_output = model(**audio_input)
        loss, _ = self.compute_loss(model, audio_output, sample, reduce=reduce)
        return loss
    
    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss
    
    def forward_ext_mt(self, model, sample, reduce):
        text_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def compute_jsd_loss(self, st_lprobs, mt_lprobs):
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum()
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss

    def get_target_masked(self, sample, mask_ratio):
        # We randomly mask target with 15% probability
        target_collapsed = sample["target"].clone().detach()
        prob = torch.ones_like(target_collapsed) * mask_ratio
        prob_mask = torch.bernoulli(prob).bool().to(target_collapsed.device)

        padding_mask = target_collapsed.eq(self.padding_idx)
        # 这里不排除<eos>
        prob_mask = prob_mask & (~padding_mask)
        # bos = 0 as <mask>
        target_collapsed.masked_fill_(prob_mask, 0)
        masked_num = prob_mask.sum().item()
        return target_collapsed, prob_mask, masked_num

    def forward_cmlm(self, model, sample, mode, target_collapsed, collasped_mask):
        if mode == "st":
            input = {
                "src_tokens": sample["net_input"]["audio"],
                "src_lengths": sample["net_input"]["audio_lengths"],
                "mode": "st",
                "prev_output_tokens": target_collapsed,
            }
        elif mode == "mt":
             input = {
                "src_tokens": sample["net_input"]["source"],
                "src_lengths": sample["net_input"]["source_lengths"],
                "mode": "mt",
                "prev_output_tokens": target_collapsed,
            }
        else:
            raise RuntimeError("mode must be st or mt")
        # self-attention no mask
        output = model.forward_cmlm(**input)

        lprobs = model.get_normalized_probs(output, log_probs=True)
        lprobs = lprobs[collasped_mask]
        target = sample["target"][collasped_mask]

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=True,
        )
        return loss, lprobs

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        st_cmlm_loss, mt_cmlm_loss, jsd_cmlm_loss = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        st_size, mt_size, ext_mt_size = 0, 0, 0
        masked_num = 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            # st + mt
            if self.mt_finetune and self.training:
                st_loss = self.forward_st(model, sample, reduce)
                mt_loss = self.forward_mt(model, sample, reduce)

                target_collapsed, collapsed_mask, masked_num = self.get_target_masked(sample, self.cmlm_mask_ratio)
                st_cmlm_loss, st_cmlm_lprobs = self.forward_cmlm(model, sample, "st", target_collapsed, collapsed_mask)
                mt_cmlm_loss, mt_cmlm_lprobs = self.forward_cmlm(model, sample, "mt", target_collapsed, collapsed_mask)
                jsd_cmlm_loss = self.compute_jsd_loss(st_cmlm_lprobs, mt_cmlm_lprobs)

                loss = st_loss + mt_loss + self.cmlm_weight*(st_cmlm_loss + mt_cmlm_loss + jsd_cmlm_loss)
                st_size = mt_size = sample_size = sample["ntokens"]
            # st(dev or train only)
            else:
                loss = st_loss = self.forward_st(model, sample, reduce)
                st_size = sample_size = sample["ntokens"]
        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "st_cmlm_loss": st_cmlm_loss.data,
            "mt_cmlm_loss": mt_cmlm_loss.data,
            "jsd_cmlm_loss": jsd_cmlm_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "masked_num": masked_num,
        }
        
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)

        st_cmlm_loss_sum = sum(log.get("st_cmlm_loss", 0) for log in logging_outputs)
        mt_cmlm_loss_sum = sum(log.get("mt_cmlm_loss", 0) for log in logging_outputs)
        jsd_cmlm_loss_sum = sum(log.get("jsd_cmlm_loss", 0) for log in logging_outputs)
        masked_num_sum = sum(log.get("masked_num", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )
        metrics.log_scalar(
            "st_cmlm_loss", st_cmlm_loss_sum / masked_num_sum / math.log(2) if masked_num_sum != 0 else 0, masked_num_sum, round=3
        )
        metrics.log_scalar(
            "mt_cmlm_loss", mt_cmlm_loss_sum / masked_num_sum / math.log(2) if masked_num_sum != 0 else 0, masked_num_sum, round=3
        )
        metrics.log_scalar(
            "jsd_cmlm_loss", jsd_cmlm_loss_sum / masked_num_sum / math.log(2) if masked_num_sum != 0 else 0, masked_num_sum, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True