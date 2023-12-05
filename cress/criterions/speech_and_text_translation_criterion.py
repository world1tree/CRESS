# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import numpy as np
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

    def compute_jsd_loss_without_pad(self, st_lprobs, mt_lprobs):
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss

    def compute_jsd_loss_part(self, st_lprobs, mt_lprobs, target, ignore_index, y_masked_info):
        pad_mask = (y_masked_info | (target.eq(ignore_index)))
        pad_mask = pad_mask.view(-1)
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        kl_loss_mt.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss

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
        kl_loss_st = F.kl_div(st_lprobs, teacher_lprobs.detach(), log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(mt_lprobs, teacher_lprobs.detach(), log_target=True, reduction="none").sum(-1)
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

        with torch.no_grad():
            all_tokens, st_correct_tokens, st_correct_matrix, st_probs = self.collect_result(model, audio_output,
                                                                                             sample)

        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)
        return loss, lprobs, target, all_tokens, st_correct_tokens, st_correct_matrix, st_probs
    
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

        with torch.no_grad():
            all_tokens, mt_correct_tokens, mt_correct_matrix, mt_probs = self.collect_result(model, decoder_out, sample)

        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        return loss, lprobs, target, all_tokens, mt_correct_tokens, mt_correct_matrix, mt_probs

    def forward_ext_mt(self, model, sample, reduce):
        text_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def forward_concat(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["concat_input"]["source"],
            "src_lengths": sample["concat_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        # 一定要用float32, 否则会溢出
        lprobs = F.log_softmax(text_output[0], dim=-1, dtype=torch.float32)
        target = sample["target"]
        selected = sample["concat_input"]["y_masked_info"]
        # only calculate loss for selected tokens
        lprobs_selected = lprobs[selected]
        target_selected = target[selected]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs_selected,
            target_selected,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, lprobs_selected, selected

    def collect_result(self, model, decoder_output, sample):
        probs = model.get_normalized_probs(decoder_output, log_probs=False)
        lprobs = model.get_normalized_probs(decoder_output, log_probs=True)
        target = model.get_targets(sample, decoder_output)
        pred = torch.argmax(lprobs, dim=-1)
        pad_mask = target.eq(self.padding_idx)
        batch_tokens = int(torch.sum(~pad_mask).item())
        correct_matrix = pred.eq(target) & (~pad_mask)
        correct_tokens = int(torch.sum(correct_matrix).item())
        return batch_tokens, correct_tokens, correct_matrix.detach(), probs.detach()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        jsd_loss = torch.Tensor([0])
        concat_loss = torch.Tensor([0])
        jsd_loss2 = torch.Tensor([0])
        st_size, mt_size, ext_mt_size = 0, 0, 0
        masked_num = 0

        mode = sample["net_input"]["mode"]
        bsz, seq_len = sample["target"].shape
        if mode == "st":
            # st + mt
            if self.mt_finetune and self.training:
                # We need average loss per token

                st_size = mt_size = sample_size = sample["ntokens"]
                st_loss, st_lprobs, st_target, st_all_tokens, st_tokens_correct, st_correct_matrix, st_probs = self.forward_st(model, sample, reduce)
                # mt_loss = self.forward_mt(model, sample, reduce)
                mt_loss, x_cross_s_lprobs, mt_target, mt_all_tokens, mt_tokens_correct, mt_correct_matrix, mt_probs = self.forward_x_cross_s(model, sample, reduce)

                assert st_all_tokens == mt_all_tokens == sample["ntokens"]
                # jsd loss between st and mt
                jsd_loss = self.compute_jsd_loss(st_lprobs, x_cross_s_lprobs, st_target, mt_target, self.padding_idx)

                # st和mt都不能预测正确的单词(可以减少mask的数量), 需要排除padding
                target_pading_mask = sample["target"].eq(self.padding_idx)
                st_mt_incorrect_matrix = (~st_correct_matrix) & (~mt_correct_matrix) & (~target_pading_mask)

                # -----------------------------------------------------
                # -----------------------------------------------------
                # -----------------------------------------------------
                # We need to modify concat_source and y_masked_info
                concat_source = sample["concat_input"]["source"]
                # print(concat_source)
                concat_lengths = sample["concat_input"]["source_lengths"]
                source_lengths = sample["net_input"]["source_lengths"]
                target_lengths = sample["target_lengths"]
                y_masked_info = sample["concat_input"]["y_masked_info"]
                assert st_mt_incorrect_matrix.shape == y_masked_info.shape
                # y_masked_info是target端的
                y_masked_info = st_mt_incorrect_matrix
                for i in range(bsz):
                    assert concat_lengths[i] - source_lengths[i] == target_lengths[i]
                    # st_mt_incorrect_matrix[i][i+1:] = False
                    concat_source[i, source_lengths[i]:concat_lengths[i]][st_mt_incorrect_matrix[i, :target_lengths[i]]] = 0 # 使用bos作为mask字符
                concat_source = concat_source.detach()
                y_masked_info = y_masked_info.detach()
                sample["concat_input"]["source"] = concat_source
                sample["concat_input"]["y_masked_info"] = y_masked_info
                # print(concat_source)
                # print(sample["net_input"]["source"])
                # -----------------------------------------------------
                # -----------------------------------------------------
                # -----------------------------------------------------
                concat_loss, concat_lprobs_selected, selected = self.forward_concat(model, sample, reduce)
                masked_num = selected.sum().item()

                st_lprobs_selected = st_lprobs.view(bsz, seq_len, -1)[selected]
                x_cross_s_lprobs_selected = x_cross_s_lprobs.view(bsz, seq_len, -1)[selected]

                # jsd1 = self.compute_jsd_loss_without_pad(st_lprobs_selected, concat_lprobs_selected)
                # jsd2 = self.compute_jsd_loss_without_pad(x_cross_s_lprobs_selected, concat_lprobs_selected)
                jsd_loss2 = self.compute_kl_loss(st_lprobs_selected, x_cross_s_lprobs_selected, concat_lprobs_selected)

                loss = ((concat_loss + jsd_loss2)/masked_num) + ((st_loss + mt_loss + jsd_loss)/sample_size)
            # st(dev or train only)
            else:
                st_size = sample_size = sample["ntokens"]
                st_loss, _, _ = self.forward_st(model, sample, reduce)
                loss = st_loss / sample_size
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
            "concat_loss": concat_loss.data,
            "jsd_loss2": jsd_loss2.data,
            "masked_num": masked_num,
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
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        jsd_loss2_sum = sum(log.get("jsd_loss2", 0) for log in logging_outputs)
        concat_loss_sum = sum(log.get("concat_loss", 0) for log in logging_outputs)
        masked_num_sum = sum(log.get("masked_num", 0) for log in logging_outputs)
        # jsd_num_sum = mt_sample_size - masked_num_sum

        metrics.log_scalar(
            "loss", loss_sum / math.log(2), 1, round=3
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
            "concat_loss", concat_loss_sum / masked_num_sum / math.log(2) if masked_num_sum != 0 else 0, masked_num_sum, round=3
        )
        metrics.log_scalar(
            "jsd_loss2", jsd_loss2_sum / masked_num_sum / math.log(2) if masked_num_sum != 0 else 0, masked_num_sum, round=3
        )
        metrics.log_scalar(
            "hard_word_percentage", masked_num_sum / sample_size if sample_size != 0 else 0, sample_size, round=3
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