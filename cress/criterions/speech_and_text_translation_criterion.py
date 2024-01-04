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

    def compute_kl_loss(self, st_lprobs, mt_lprobs, teacher_lprobs_st, teacher_lprobs_mt):
        kl_loss_st = F.kl_div(st_lprobs, teacher_lprobs_st, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(mt_lprobs, teacher_lprobs_mt, log_target=True, reduction="none").sum(-1)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        return kl_loss_st, kl_loss_mt

    def get_masked_idx(self, probs, sample, eps=0.2):
        target = sample["target"].unsqueeze(-1)
        # res[i][j][k] = lprobs[i][j][target[i][j][k]]
        hypo = probs.gather(dim=-1, index=target).squeeze(-1)
        masked_idx = (hypo < eps)
        # 排除掉padding[1], eos[2]
        padding_mask = (sample["target"].eq(self.padding_idx)) | (sample["target"].eq(2))
        masked_idx = masked_idx & (~padding_mask)
        return masked_idx

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_output = model(**audio_input)
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)

        probs = F.softmax(audio_output[0], dim=-1, dtype=torch.float32)
        audio_masked_idx = self.get_masked_idx(probs, sample)
        return loss, lprobs, target, audio_masked_idx
    
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

    def forward_cmlm(self, model, sample, audio_masked_idx, text_masked_idx):
        target_audio = sample["target"].clone().detach()
        target_audio.masked_fill_(audio_masked_idx, 0)
        target_text = sample["target"].clone().detach()
        target_text.masked_fill_(text_masked_idx, 0)

        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
        }
        model.cmlm_model[0] = model.cmlm_model[0].to(sample["target"].device)
        model.cmlm_model[1] = model.cmlm_model[1].to(sample["target"].device)
        with torch.no_grad():
            encoder_out = model.cmlm_model[0](**text_input)
            decoder_out_st = model.cmlm_model[1](prev_output_tokens=target_audio, encoder_out=encoder_out, full_context_alignment=True)
            lprobs_st = model.get_normalized_probs(decoder_out_st, log_probs=True)
            lprobs_st = lprobs_st[audio_masked_idx]

            decoder_out_mt = model.cmlm_model[1](prev_output_tokens=target_text, encoder_out=encoder_out, full_context_alignment=True)
            lprobs_mt = model.get_normalized_probs(decoder_out_mt, log_probs=True)
            lprobs_mt = lprobs_mt[text_masked_idx]

        return lprobs_st.detach(), lprobs_mt.detach()

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


        probs = F.softmax(decoder_out[0], dim=-1, dtype=torch.float32)
        text_masked_idx = self.get_masked_idx(probs, sample)
        return loss, lprobs, target, text_masked_idx

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
        kl_loss_st = torch.Tensor([0])
        kl_loss_mt = torch.Tensor([0])
        st_size, mt_size, ext_mt_size = 0, 0, 0
        masked_num_st = 0
        masked_num_mt = 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            # st + mt
            if self.mt_finetune and self.training:
                st_size = mt_size = sample_size = sample["ntokens"]
                bsz, seq_len = sample["target"].size()
                st_loss, st_lprobs, st_target, audio_masked_idx = self.forward_st(model, sample, reduce)
                # mt_loss = self.forward_mt(model, sample, reduce)
                mt_loss, x_cross_s_lprobs, mt_target, text_masked_idx = self.forward_x_cross_s(model, sample, reduce)
                # cmlm_loss的权重可以调整
                cmlm_lprobs_st, cmlm_probs_mt = self.forward_cmlm(model, sample, audio_masked_idx, text_masked_idx)
                jsd_loss = self.compute_jsd_loss(st_lprobs, x_cross_s_lprobs, st_target, mt_target, self.padding_idx)
                masked_num_st = audio_masked_idx.sum().item()
                masked_num_mt = text_masked_idx.sum().item()
                st_lprobs_selected = st_lprobs.view(bsz, seq_len, -1)[audio_masked_idx]
                x_cross_s_lprobs_selected = x_cross_s_lprobs.view(bsz, seq_len, -1)[text_masked_idx]
                kl_loss_st, kl_loss_mt = self.compute_kl_loss(st_lprobs_selected, x_cross_s_lprobs_selected, cmlm_lprobs_st, cmlm_probs_mt)
                loss = (st_loss + mt_loss + jsd_loss)/sample_size + (kl_loss_st/masked_num_st) + (kl_loss_mt/masked_num_mt)
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
            "jsd_loss": jsd_loss.data,
            "kl_loss_st": kl_loss_st.data,
            "kl_loss_mt": kl_loss_mt.data,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "masked_num_st": masked_num_st,
            "masked_num_mt": masked_num_mt,
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
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        kl_loss_st_sum = sum(log.get("kl_loss_st", 0) for log in logging_outputs)
        kl_loss_mt_sum = sum(log.get("kl_loss_mt", 0) for log in logging_outputs)
        masked_num_st = sum(log.get("masked_num_st", 0) for log in logging_outputs)
        masked_num_mt = sum(log.get("masked_num_mt", 0) for log in logging_outputs)

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
            "kl_loss_st", kl_loss_st_sum / masked_num_st / math.log(2) if masked_num_st != 0 else 0, masked_num_st, round=3
        )
        metrics.log_scalar(
            "kl_loss_mt", kl_loss_mt_sum / masked_num_mt / math.log(2) if masked_num_mt != 0 else 0, masked_num_mt, round=3
        )
        metrics.log_scalar(
            "jsd_loss", jsd_loss_sum / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "low_st", masked_num_st / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "low_mt", masked_num_mt / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
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
