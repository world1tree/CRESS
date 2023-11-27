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

    def compute_l1_loss(self, lprobs1, lprobs2, target, ignore_index):
        l1_loss = F.l1_loss(lprobs1, lprobs2, reduction="none")
        pad_mask = target.eq(ignore_index).unsqueeze(-1)
        l1_loss.masked_fill_(pad_mask, 0.0)
        l1_loss = l1_loss.sum()
        return l1_loss

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_output = model(**audio_input)
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)
        return loss, lprobs, target

    def forward_st_encoder(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
        }
        # audio_output = model(**audio_input)
        encoder_out = model.encoder(**audio_input)
        return encoder_out

    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, text_output, sample, reduce=reduce)
        return loss, lprobs, target

    def forward_mt_encoder(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
        }
        encoder_out = model.encoder(**text_input)
        return encoder_out

    def forward_x_cross_s(self, model, sample, mt_encoder, reduce):
        text_input = {
            "audio": sample["net_input"]["audio"],
            "audio_lengths": sample["net_input"]["audio_lengths"],
            "source": sample["net_input"]["source"],
        }
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        x_cross_s_encoder = model.encoder.forward_x_cross_s(**text_input)

        # B, T
        encoder_out_padding_mask = mt_encoder["encoder_padding_mask"][0]
        encoder_embedding = mt_encoder["encoder_embedding"][0]

        # B, T, D
        encoder_out_origin = mt_encoder["encoder_out"][0].transpose(0, 1)
        x_cross_s_encoder_out = x_cross_s_encoder["encoder_out"][0].transpose(0, 1)

        # get random mix ratio
        bsz, text_len, emb_dim = encoder_out_origin.size()
        # p = 0.5 means 50% of the time we use encoder_out_origin
        probability_matrix = torch.full((bsz, ), 0.5, device=encoder_out_origin.device, dtype=encoder_out_origin.dtype)
        selected_index = torch.bernoulli(probability_matrix).bool().to(encoder_out_origin.device)

        # get mix encoder_out
        mix_encoder_out = torch.zeros((bsz, text_len, emb_dim), device=encoder_out_origin.device, dtype=encoder_out_origin.dtype)
        mix_encoder_out[selected_index] = encoder_out_origin[selected_index]
        mix_encoder_out[~selected_index] = x_cross_s_encoder_out[~selected_index]
        mix_encoder_out = mix_encoder_out.transpose(0, 1)

        encoder_out = {
            "encoder_out": [mix_encoder_out],  # T x B x C
            "encoder_padding_mask": [encoder_out_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        decoder_out = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        return loss, lprobs, target

    def forward_s_cross_x(self, model, sample, st_encoder, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["source"],
            "audio": sample["net_input"]["audio"],
            "audio_lengths": sample["net_input"]["audio_lengths"],
        }
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        s_cross_x_encoder = model.encoder.forward_s_cross_x(**audio_input)

        # B, T
        encoder_out_padding_mask = st_encoder["encoder_padding_mask"][0]
        encoder_embedding = st_encoder["encoder_embedding"][0]

        # B, T, D
        encoder_out_origin = st_encoder["encoder_out"][0].transpose(0, 1)
        s_cross_x_encoder_out = s_cross_x_encoder["encoder_out"][0].transpose(0, 1)

        # get random mix ratio
        bsz, text_len, emb_dim = encoder_out_origin.size()
        # p = 0.5 means 50% of the time we use encoder_out_origin
        probability_matrix = torch.full((bsz, ), 0.5, device=encoder_out_origin.device, dtype=encoder_out_origin.dtype)
        selected_index = torch.bernoulli(probability_matrix).bool().to(encoder_out_origin.device)

        # get mix encoder_out
        mix_encoder_out = torch.zeros((bsz, text_len, emb_dim), device=encoder_out_origin.device, dtype=encoder_out_origin.dtype)
        mix_encoder_out[selected_index] = encoder_out_origin[selected_index]
        mix_encoder_out[~selected_index] = s_cross_x_encoder_out[~selected_index]
        mix_encoder_out = mix_encoder_out.transpose(0, 1)

        encoder_out = {
            "encoder_out": [mix_encoder_out],  # T x B x C
            "encoder_padding_mask": [encoder_out_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        decoder_out = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        return loss, lprobs, target

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
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        jsd_loss = torch.Tensor([0])
        s_cross_x_loss = torch.Tensor([0])

        st_size, mt_size, ext_mt_size = 0, 0, 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            # st + mt
            if self.mt_finetune and self.training:
                # st_loss, st_lprobs, st_target = self.forward_st(model, sample, reduce)
                st_encoder = self.forward_st_encoder(model, sample, reduce)
                st_loss, st_lprobs, _ = self.forward_s_cross_x(model, sample, st_encoder, reduce)
                mt_encoder = self.forward_mt_encoder(model, sample, reduce)
                mt_loss, mt_lprobs, mt_target = self.forward_x_cross_s(model, sample, mt_encoder, reduce)

                jsd_loss = self.compute_jsd_loss(st_lprobs, mt_lprobs, mt_target, mt_target, self.padding_idx)
                loss = st_loss + mt_loss + jsd_loss
                st_size = mt_size = sample_size = sample["ntokens"]
            # st(dev or train only)
            else:
                st_loss, _, _ = self.forward_st(model, sample, reduce)
                loss = st_loss
                st_size = sample_size = sample["ntokens"]
        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "s_cross_x_loss": s_cross_x_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "jsd_loss": jsd_loss.data,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        s_cross_x_loss_sum = sum(log.get("s_cross_x_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "s_cross_x_loss", s_cross_x_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "jsd_loss", jsd_loss_sum / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True