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
    label_smoothed_nll_loss
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

    def compute_kl_loss(self, st_lprobs, mt_lprobs, teacher_lprobs):
        kl_loss_st = F.kl_div(st_lprobs, teacher_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(mt_lprobs, teacher_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = kl_loss_st + kl_loss_mt
        return kl_loss
    
    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_output = model(**audio_input)
        loss, _, _, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)

        lprobs = F.log_softmax(audio_output[0][:,:,model.vocab_padding_mask], dim=-1)
        return loss, lprobs, target
    
    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        loss, _, _, target = self.compute_loss_with_lprobs(model, text_output, sample, reduce=reduce)
        lprobs = F.log_softmax(text_output[0][:,:,model.vocab_padding_mask], dim=-1)
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
        loss, _, _, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        lprobs = F.log_softmax(decoder_out[0][:,:,model.vocab_padding_mask], dim=-1)
        return loss, lprobs, target

    def forward_masked_lm(self, model, sample, reduce):
        bert_model = model.bert_model
        bert_input = sample["bert_input"]
        # labels = sample["bert_labels"]
        # B, T, 10000
        with torch.no_grad():
            logits = bert_model(**bert_input).logits
            # this is masked_indices
            # mask_indices = labels.ne(-100)
            mask_indices = bert_input["input_ids"].eq(103) & bert_input["token_type_ids"].eq(1)
            masked_logits = logits[mask_indices]
            # masked_logits = torch.log_softmax(masked_logits, dim=-1)
            masked_logits = F.log_softmax(masked_logits[:,model.vocab_padding_mask], dim=-1)

        return masked_logits.detach()

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
        masked_lm_loss = torch.Tensor([0])
        jsd_loss = torch.Tensor([0])
        kl_loss = torch.Tensor([0])
        st_size, mt_size, ext_mt_size = 0, 0, 0
        masked_num = 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            # st + mt
            bsz, tsz = sample["target"].size()
            if self.mt_finetune and self.training:
                st_loss, st_lprobs, st_target = self.forward_st(model, sample, reduce)
                # mt_loss = self.forward_mt(model, sample, reduce)
                mt_loss, x_cross_s_lprobs, mt_target = self.forward_mt(model, sample, reduce)
                # mt_loss, x_cross_s_lprobs, mt_target = self.forward_x_cross_s(model, sample, reduce)

                # use bert as teacher, use st and mt as student
                student1_logits = x_cross_s_lprobs
                student1_logits = student1_logits[sample["y_masked_info"]]
                student2_logits = st_lprobs
                student2_logits = student2_logits[sample["y_masked_info"]]
                masked_num = sample["y_masked_info"].sum().item()
                teacher_logits = self.forward_masked_lm(model, sample, reduce)
                kl_loss_st = F.kl_div(student1_logits, teacher_logits, log_target=True, reduction="none").sum(-1)
                kl_loss_mt = F.kl_div(student2_logits, teacher_logits, log_target=True, reduction="none").sum(-1)
                # 这里是否需要除以2有待商榷
                kl_loss = (kl_loss_st.sum() + kl_loss_mt.sum()) / 2
                # 这里应该使用全部的jsd
                # jsd_loss = self.compute_jsd_loss(st_lprobs, x_cross_s_lprobs, st_target, mt_target, self.padding_idx)
                st_size = mt_size = sample_size = sample["ntokens"]
                loss = ((st_loss + mt_loss)/sample_size) + kl_loss/masked_num
            # st(dev or train only)
            else:
                st_loss, _, _ = self.forward_st(model, sample, reduce)
                loss = st_loss
                # masked_lm_loss, masked_num = self.forward_masked_lm(model, sample, reduce)
                # loss = masked_lm_loss
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
            "masked_lm_loss": masked_lm_loss.data,
            "masked_num": masked_num,
            "jsd_loss": jsd_loss.data,
            "kl_loss": kl_loss.data,
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
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        masked_lm_loss_sum = sum(log.get("masked_lm_loss", 0) for log in logging_outputs)
        masked_num_sum = sum(log.get("masked_num", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "masked_lm_loss", masked_lm_loss_sum / math.log(2), 1, round=3
        )
        metrics.log_scalar(
            "jsd_loss", jsd_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "kl_loss", kl_loss_sum / masked_num_sum / math.log(2) if masked_num_sum != 0 else 0, masked_num_sum, round=3
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