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
    label_smoothed_nll_loss,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechAndTextTranslationCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mt_finetune: bool = field(
        default=False,
        metadata={"help": "st + mt multi-task finetune"},
    )

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
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.mt_finetune = mt_finetune

    def compute_loss_with_lprobs(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target

    def compute_jsd_loss(self, st_lprobs, mt_lprobs, st_target, mt_target, ignore_index):
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum(-1)
        pad_mask = st_target.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        pad_mask = mt_target.eq(ignore_index)
        kl_loss_mt.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss

    def compute_kl_loss(self, st_lprobs, mt_lprobs, concat_lprobs, target, ignore_index):
        kl_loss_st = F.kl_div(st_lprobs, concat_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(mt_lprobs, concat_lprobs, log_target=True, reduction="none").sum(-1)
        pad_mask = target.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        kl_loss_mt.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = kl_loss_st + kl_loss_mt
        return kl_loss

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
        }
        encoder_out = model.encoder(**audio_input)
        encoder_audio_output = encoder_out["encoder_out"][0]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        audio_output = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)
        return loss, lprobs, target, encoder_audio_output
    
    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
        }
        encoder_out = model.encoder(**text_input)
        encoder_text_output = encoder_out["encoder_out"][0]
        # prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        # text_output = model.decoder(
        #     prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        # )
        # loss, _, lprobs, target = self.compute_loss_with_lprobs(model, text_output, sample, reduce=reduce)
        # return loss, lprobs, target, encoder_text_output
        return encoder_text_output

    def forward_x_cross_s(self, model, sample, reduce, encoder_x_output):
        text_input = {
            "audio": sample["net_input"]["audio"],
            "audio_lengths": sample["net_input"]["audio_lengths"],
            "source": sample["net_input"]["source"],
        }
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        encoder_out = model.encoder.forward_x_cross_s(**text_input)
        encoder_x_cross_s_output = encoder_out["encoder_out"][0]
        # simple way to merge text and audio modality, g代表参考多少文本
        g = model.encoder.self_attn_gate(torch.cat([encoder_x_output, encoder_x_cross_s_output], dim=-1))
        encoder_mixmodal = encoder_x_output*g + encoder_x_cross_s_output*(1-g)
        encoder_out["encoder_out"][0] = encoder_mixmodal

        decoder_out = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        return loss, lprobs, target, g

    def forward_ext_mt(self, model, sample, reduce):
        text_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.])
        mix_loss, jsd_loss = torch.Tensor([0.]), torch.Tensor([0.])
        text_percent = torch.Tensor([0.])
        st_size, mt_size, ext_mt_size = 0, 0, 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            # st + mt
            if self.mt_finetune and self.training:
                st_loss, st_lprobs, _, encoder_audio_output = self.forward_st(model, sample, reduce)
                # mt_loss, mt_lprobs, _, encoder_text_output = self.forward_mt(model, sample, reduce)
                encoder_text_output = self.forward_mt(model, sample, reduce)
                mix_loss, mix_lprobs, target, g = self.forward_x_cross_s(model, sample, reduce, encoder_text_output)
                text_percent = torch.mean(g)
                jsd_loss = self.compute_jsd_loss(st_lprobs, mix_lprobs, target, target, self.padding_idx)
                loss = st_loss + mix_loss + jsd_loss
                st_size = mt_size = sample_size = sample["ntokens"]
            # st(dev or train only)
            else:
                st_loss, _, _, _ = self.forward_st(model, sample, reduce)
                loss = st_loss
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
            "mix_loss": mix_loss.data,
            "jsd_loss": jsd_loss.data,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "text_percent": text_percent.data,
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
        mix_loss_sum = sum(log.get("mix_loss", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        text_percent_sum = sum(log.get("text_percent", 0) for log in logging_outputs)

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
            "mix_loss", mix_loss_sum / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "jsd_loss", jsd_loss_sum / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )
        metrics.log_scalar(
            "text_percent", text_percent_sum, 1, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
