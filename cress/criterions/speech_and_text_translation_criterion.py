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

    def compute_target_distribute_of_x_cross_s_no_grad(self, model, sample):
        text_audio_input = {
            "audio": sample["net_input"]["audio"],
            "audio_lengths": sample["net_input"]["audio_lengths"],
            "source": sample["net_input"]["source"],
        }
        x_cross_s_encoder_out = model.encoder.forward_x_cross_s(**text_audio_input)
        # decoder此时是没有梯度的
        with torch.no_grad():
            prev_output_tokens = sample["net_input"]["prev_output_tokens"]
            decoder_out = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=x_cross_s_encoder_out
            )
            lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
            target = model.get_targets(sample, decoder_out)
        return lprobs, target, x_cross_s_encoder_out

    def compute_target_distribute_of_x_no_grad(self, model, sample):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
        }
        mt_encoder_output = model.encoder(**text_input)
        with torch.no_grad():
            prev_output_tokens = sample["net_input"]["prev_output_tokens"]
            decoder_out = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=mt_encoder_output
            )
            lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
            target = model.get_targets(sample, decoder_out)
        return lprobs, target, mt_encoder_output

    def forward_x_cross_s_mix_auto(self, model, sample, reduce):
        # target1与target2应该相同
        y1_lprobs, target1, x_cross_s_encoder = self.compute_target_distribute_of_x_cross_s_no_grad(model, sample)
        y2_lprobs, target2, x_encoder = self.compute_target_distribute_of_x_no_grad(model, sample)
        p1_lprobs = y1_lprobs.gather(dim=-1, index=target1.unsqueeze(-1))
        p2_lprobs = y2_lprobs.gather(dim=-1, index=target2.unsqueeze(-1))
        # B, T
        p1_lprobs = p1_lprobs.squeeze(-1)
        p2_lprobs = p2_lprobs.squeeze(-1)
        # mask
        pad_mask1 = target1.eq(self.padding_idx)
        p1_lprobs.masked_fill_(pad_mask1, 0.0)
        pad_mask2 = target2.eq(self.padding_idx)
        p2_lprobs.masked_fill_(pad_mask2, 0.0)
        # sum: p1表示参考语音的，p2表示没有参考语音
        p1_sum_lprobs = p1_lprobs.sum(dim=-1)
        p2_sum_lprobs = p2_lprobs.sum(dim=-1)
        # 0表示选择参考语音，1表示选择参考文本
        label = torch.argmax(torch.stack([p1_sum_lprobs, p2_sum_lprobs]), dim=0)
        selected_index = label.bool().detach()

        encoder_out_attend_s = x_cross_s_encoder["encoder_out"][0].transpose(0, 1)
        # get basic shape
        bsz, text_len, emb_dim = encoder_out_attend_s.size()
        # x_cross_s is better
        bsz_x_cross_s = torch.sum(~selected_index).float()
        encoder_out_padding_mask = x_cross_s_encoder["encoder_padding_mask"][0]
        encoder_embedding = x_cross_s_encoder["encoder_embedding"][0]

        encoder_out_origin = x_encoder["encoder_out"][0].transpose(0, 1)

        mix_encoder_out = torch.zeros((bsz, text_len, emb_dim), device=encoder_out_origin.device, dtype=encoder_out_origin.dtype)
        mix_encoder_out[selected_index] = encoder_out_origin[selected_index]
        mix_encoder_out[~selected_index] = encoder_out_attend_s[~selected_index]
        mix_encoder_out = mix_encoder_out.transpose(0, 1)

        encoder_out = {
            "encoder_out": [mix_encoder_out],  # T x B x C
            "encoder_padding_mask": [encoder_out_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        decoder_out = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        return loss, lprobs, target, bsz_x_cross_s

    def forward_x_cross_s_mix(self, model, sample, reduce):
        # x_cross_s
        text_audio_input = {
            "audio": sample["net_input"]["audio"],
            "audio_lengths": sample["net_input"]["audio_lengths"],
            "source": sample["net_input"]["source"],
        }
        x_cross_s_encoder_out = model.encoder.forward_x_cross_s(**text_audio_input)
        # B, T, D
        encoder_out_attend_s = x_cross_s_encoder_out["encoder_out"][0].transpose(0, 1)
        # B, T
        encoder_out_padding_mask = x_cross_s_encoder_out["encoder_padding_mask"][0]
        encoder_embedding = x_cross_s_encoder_out["encoder_embedding"][0]

        # normal mt
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
        }
        mt_encoder_ouput = model.encoder(**text_input)
        # B, T, D
        encoder_out_origin = mt_encoder_ouput["encoder_out"][0].transpose(0, 1)

        # get random mix ratio
        bsz, text_len, emb_dim = encoder_out_origin.size()
        # p = 0.5 means 50% of the time we use encoder_out_origin
        probability_matrix = torch.full((bsz, ), 0.3, device=encoder_out_origin.device)
        selected_index = torch.bernoulli(probability_matrix).bool().to(encoder_out_origin.device)

        # get mix encoder_out
        mix_encoder_out = torch.zeros((bsz, text_len, emb_dim), device=encoder_out_origin.device)
        mix_encoder_out[selected_index] = encoder_out_origin[selected_index]
        mix_encoder_out[~selected_index] = encoder_out_attend_s[~selected_index]
        mix_encoder_out = mix_encoder_out.transpose(0, 1)

        encoder_out = {
            "encoder_out": [mix_encoder_out],  # T x B x C
            "encoder_padding_mask": [encoder_out_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        decoder_out = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        return loss, lprobs, target

    def forward_x_cross_s(self, model, sample, reduce):
        text_input = {
            "audio": sample["net_input"]["audio"],
            "audio_lengths": sample["net_input"]["audio_lengths"],
            "source": sample["net_input"]["source"],
        }
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        encoder_out = model.encoder.forward_x_cross_s(**text_input)
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
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        jsd_loss = torch.Tensor([0]).cuda()
        st_size, mt_size, ext_mt_size = 0, 0, 0
        bsz_x_cross_s = torch.tensor(0.)
        batch_size = 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            # st + mt
            if self.mt_finetune and self.training:
                st_loss, st_lprobs, st_target = self.forward_st(model, sample, reduce)
                # mt_loss = self.forward_mt(model, sample, reduce)
                mt_loss, x_cross_s_lprobs, mt_target, bsz_x_cross_s = self.forward_x_cross_s_mix_auto(model, sample, reduce)
                jsd_loss = self.compute_jsd_loss(st_lprobs, x_cross_s_lprobs, st_target, mt_target, self.padding_idx)
                loss = st_loss + mt_loss + jsd_loss
                st_size = mt_size = sample_size = sample["ntokens"]
                batch_size = sample["target"].size(0)
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
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "jsd_loss": jsd_loss.data,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "bsz_x_cross_s": bsz_x_cross_s.data,
            "sample_size": sample_size,
            "batch_size": batch_size,
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
        bsz_x_cross_s = sum(log.get("bsz_x_cross_s", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        batch_size = sum(log.get("batch_size", 0) for log in logging_outputs)
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
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "jsd_loss", jsd_loss_sum / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )
        metrics.log_scalar(
            "bsz_x_cross_s", bsz_x_cross_s / batch_size if batch_size != 0 else 0, batch_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True